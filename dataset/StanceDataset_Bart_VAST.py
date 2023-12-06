import os
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from utils.rich_tqdm import progress
from utils.get_bert_and_tokenizer import getTokenizer

class StanceData(Dataset):
    def __init__(self,args,file_path,wiki_path) :
        super().__init__()
        self.logger=args['logger']
        self.args=args

        self.tokenizer=getTokenizer(
            self.logger,
            
        )