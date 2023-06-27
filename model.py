import os
import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification, AutoModelForCausalLM, AutoTokenizer

class VisionLM(nn.Module):
    def __init__(self, processor, vit, llm, tokenizer, freeze_vit=True):
        super().__init__()

        self.tokenizer = tokenizer
        self.llm = llm
        self.processor = processor
        self.vit = vit
        if self.vit.config.hidden_size != self.llm.config.hidden_size:
            print(f'ViT hidden size {self.vit.config.hidden_size} != LLM word_embed_proj_dim {self.llm.config.hidden_size}. Adding a linear layer to map ViT hidden size to LLM word_embed_proj_dim')
            self.mapping_layer = True
            self.mapping = nn.Sequential(
                nn.Linear(self.vit.config.hidden_size, self.llm.config.word_embed_proj_dim),
            )
        else:
            self.mapping_layer = False
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.imagehere_token = '<ImageHere>'
        self.prompt = f'Image: {tokenizer.eos_token}{self.imagehere_token}{tokenizer.eos_token}'
        self.eostoken = self.tokenizer.eos_token

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split(self.imagehere_token)
            p_before_tokens = self.tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.get_llm_embeds(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.get_llm_embeds(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def get_llm_embeds(self, tokens):
        if self.llm.config.model_type == 'gpt2':
            return self.llm.transformer.wte(tokens)
        elif self.llm.config.model_type == 'gpt_neo':
            return self.llm.transformer.wte.weight[tokens]
        elif self.llm.config.model_type == 'llama':
            return self.llm.base_model.embed_tokens(tokens)
        elif self.llm.config.model_type == 'opt':
            return self.llm.model.decoder.embed_tokens(tokens)
        else:
            raise NotImplementedError
        
    def encode_image(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.vit(inputs.pixel_values.to(img.device))
        return outputs.last_hidden_state if not self.mapping_layer else self.mapping(outputs.last_hidden_state)

    def forward(self,
                image: torch.tensor,
                prompt_tokens: torch.tensor,
                attention_mask: torch.tensor,
                ):
        
        img_embeds = self.encode_image(image)
        image_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)
        
        img_embeds, atts_img = self.prompt_wrap(img_embeds, image_atts, self.prompt)
        
        to_regress_tokens = prompt_tokens
        
        targets = to_regress_tokens.masked_fill(
            to_regress_tokens == self.tokenizer.pad_token_id, -100
        )
        
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                    dtype=torch.long).to(image.device).fill_(-100)
        )
        
        targets = torch.cat([empty_targets, targets], dim=1)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.dtype,
                        device=to_regress_tokens.device) * self.tokenizer.bos_token_id
        bos_embeds = self.get_llm_embeds(bos)
        atts_bos = atts_img[:, :1]
        
        to_regress_embeds = self.get_llm_embeds(to_regress_tokens)

        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)

        attention_mask = torch.cat([atts_bos, atts_img, attention_mask], dim=1)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True
        )
        return outputs
        
    def generate(self, image, prompt_tokens, attention_mask, max_len=125, **args):
        
        img_embeds = self.encode_image(image)
        image_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        img_embeds, atts_img = self.prompt_wrap(img_embeds, image_atts, self.prompt)

        to_regress_tokens = prompt_tokens

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.dtype,
                        device=to_regress_tokens.device) * self.tokenizer.bos_token_id
        bos_embeds = self.get_llm_embeds(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.get_llm_embeds(to_regress_tokens)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_len+197,
            **args,
        )
        return outputs
    
    def from_pretrained(self, *args, **kwargs):
        self.llm = self.llm.from_pretrained(*args, **kwargs)
        return self

    def save_pretrained(self, model_name='VisionGPT', save_only_llm=True):
        if not os.path.exists(model_name):
            os.makedirs(model_name)
        if save_only_llm:
            torch.save(self.llm.state_dict(), os.path.join(model_name, '-llm.pt'))
        if self.mapping_layer:
            torch.save(self.mapping_layer.state_dict(), os.path.join(model_name, '-mapping.pt'))
        if not save_only_llm:
            torch.save(self.state_dict(), os.path.join(model_name, '-full.pt'))

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + nontrainable_params

    # Convert total_params to a human-readable format (e.g. '125M' or '1.3B')
    if total_params / 1e9 > 1:
        total_params_str = f'{total_params / 1e9:.1f}B'
    else:
        total_params_str = f'{total_params / 1e6:.1f}M'

    print(f'Number of trainable parameters: {trainable_params}')
    print(f'Number of non-trainable parameters: {nontrainable_params}')
    print(f'Total number of parameters: {total_params_str}')


def get_model(llm_id="facebook/opt-350m", vit_id='google/vit-base-patch16-224', device='cpu', tokenizer=None):
    llm = AutoModelForCausalLM.from_pretrained(llm_id, low_cpu_mem_usage=True).to(device)
    processor = ViTImageProcessor.from_pretrained(vit_id)
    vit_model = ViTForImageClassification.from_pretrained(vit_id).to(device)
    vit_base = vit_model.vit
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(llm_id)
    vision_model = VisionLM(processor=processor, llm=llm, vit=vit_base, tokenizer=tokenizer).to(device)
    count_parameters(vision_model)
    return vision_model
