import math
import torch
from PIL import Image
from io import BytesIO
from base64 import b64decode
import numpy as np
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset

class LLavaDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        print("Loading data...")

        if args.data_use == "default":
            self.ds_names = ["msvd-qa", "coco", "textcap", "image-paragraph-captioning", "coco-goi"]
            self.datasets_list = [load_dataset("MMInstruction/M3IT", ds_name, split='train') for ds_name in self.ds_names]
            self.dataset = concatenate_datasets(self.datasets_list)
        
        else:
            self.ds_names = args.data_use
            self.dataset = load_dataset("MMInstruction/M3IT", self.ds_names, split='train')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        instruction = self.dataset[index]["instruction"]
        inputs = self.dataset[index]["inputs"]
        outputs = self.dataset[index]["outputs"]
        conv = f"{instruction}\nInput: {inputs}\nOutput: {outputs}{self.tokenizer.eos_token}"
        conv = self.tokenizer(conv, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        image_base64_str_list = self.dataset[index]["image_base64_str"]
        img = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert('RGB').resize((224, 224))
        img = torch.from_numpy(np.array(img)).reshape(3, 224, 224)

        return img, conv['input_ids'].reshape(-1), conv['attention_mask'].reshape(-1)
    
    def get_loader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.args.batch_size, num_workers=self.args.num_worker, pin_memory=True)
