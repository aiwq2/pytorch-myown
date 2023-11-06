from sklearn.model_selection import train_test_split
from torch.utils.data import random_split



def dataset_split_sklearn(train_data,train_target,test_size=0.2):
    X_train,X_text,y_train,y_test=train_test_split(train_data,train_target,test_size=test_size,stratify=train_target)
    return X_train,X_text,y_train,y_test




def dataset_split_torch(dataset,test_size=0.2):
    test_num=int(len(dataset)*test_size)
    train_dataset,test_dataset=random_split(dataset,[len(dataset)-test_num,test_num])
    return train_dataset,test_dataset