import os
import torch
import json
from torch.utils.data import Dataset
from typing import Dict
from utils.rich_tqdm import progress
from utils.get_bert_and_tokenizer import getTokenizer
import numpy as np


class LineByLineTextDataset(Dataset):
    def __init__(self,args,file_path,max_length) :
        super(LineByLineTextDataset,self).__init__()
        logger=args['logger']
        tokenizer=getTokenizer(
            args['logger'],
            os.path.join(
                args['model_path']['pretrained_model_dir'],
                args['model_path']['tokenizer']
            )
        )

        if not os.path.isfile(file_path):
            logger.error(f'file_path:{file_path} not found')
            return
        
        logger.info(f'create feature from dataset file at {file_path}')

        with open(file_path, encoding="utf-8") as f:
            text_lines = [json.loads(line) for line in f.readlines()]
            texts = [line["abstract"] for line in text_lines]
            try:
                labels = [line["label_id"] for line in text_lines]
            except:
                logger.warning("Missing Data Label")
                labels = [-1 for line in text_lines]

        batch_encoding=tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

        self.dataset_len=len(texts)
        rich_dataset_progress=progress.add_task(description='prepare dataset',total=self.dataset_len)
        if args['device']=='CPU' or (args["device"] != "CPU" and args["device"] == args["global_device"]):
            progress.start()

        self.data=[]
        for i in range(self.dataset_len):
            self.data.append(
                (
                    {
                        "input_ids": torch.tensor(
                            batch_encoding["input_ids"][i], dtype=torch.long
                        ),
                        "token_type_ids": torch.tensor(
                            batch_encoding["token_type_ids"][i], dtype=torch.long
                        ),
                        "attention_mask":torch.tensor(
                            batch_encoding['attention_mask'][i],dtype=torch.long
                        ),
                        "labels": torch.tensor(labels[i], dtype=torch.long),                        
                    },
                    {}
                )
            )
            progress.advance(rich_dataset_progress,advance=1)
        progress.stop()
        progress.remove_task(rich_dataset_progress)


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i):
        return self.data[i]


