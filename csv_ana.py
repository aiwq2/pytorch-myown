import os
import pandas as pd


file_name='score_delta.csv'

df=pd.read_csv(file_name)
label_0=df[df['label']==0]
label_1=df[df['label']==1]
label_other=df[df['label']==-1]
print(label_0.describe())
print(label_1.describe())
print(label_other.describe())