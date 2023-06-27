import argparse
from typing import Optional
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from model import get_model
from prepare_dataset import LLavaDataset

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # image, prompt, attention = inputs
        image, input_ids, attention = inputs
        # outputs = model(image, prompt, attention_mask=attention)
        outputs = model(image, input_ids, attention)
        output = outputs.logits
        loss = outputs.loss
        return (loss, output) if return_outputs else loss
    
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_id", type=str, default="facebook/opt-350m")
    parser.add_argument("--vit_id", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_use", type=str, default='default')
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = get_model(
        llm_id=args.llm_id,
        vit_id=args.vit_id,
        tokenizer=tokenizer,
        device='cpu'
    )
    
    train_dataset = LLavaDataset(args, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        logging_dir=f'{args.output_dir}-log',
        logging_steps=10,
        seed=args.seed,
        save_steps=500,
        save_total_limit=3,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to='wandb',
        learning_rate=args.lr,
    )
    
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
