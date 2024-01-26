from utils.params import config_args
import os
from loguru import logger
from utils.rich_tqdm import progress
from trainer import Trainer
from utils.get_bert_and_tokenizer import getBert
from torch.distributed import destroy_process_group
from dataset.MNIST import MNIST_Test_data,MNIST_Train_data
from dataset.BlUR.BlurDataset import *
from model.Resnet import *
from model.VIT import VIT
from model.CNN import LinearNet
from model.general.model_init import xavier_init_model,kaiming_init_model
from utils.dataset_split import dataset_split_sklearn,dataset_split_torch
import sys







if __name__=='__main__':

    # print(os.environ['MASTER_ADDR'])
    # print(os.environ['MASTER_PORT'])
    # print(os.environ['RANK'])
    # print(os.environ['LOCAL_RANK'])
    # print(os.environ['WORLD_SIZE'])
    
    # 如果要使用机器学习模型例如GBDT，请移步machine_learning_method/train.py使用
    args=config_args()
    for k,v in sorted(args.items(),key=lambda x:x[0]):
        args['logger'].info(f'{k}:{v}')

    # 自定义一些操作
    with open('score_delta.txt','w') as out:
        pass
    if os.path.exists('score_delta.csv'):
        args['logger'].info('score_delta exists,remove score_delta.csv')
        os.remove('score_delta.csv')


    # 定义dataset
    train_dataset=BlurSingle(mode='train')
    eval_dataset=BlurSingle(mode='eval')
    predict_dataset=BlurPair(mode='predict')

    # 划分训练集和验证集
    if args['mode']!='predict' and args['split_dataset'] and (isinstance(eval_dataset,list) and len(eval_dataset)==0):
        train_dataset,eval_dataset=dataset_split_torch(train_dataset,test_size=args['split_test_ratio'])
    
    # 保存训练集和验证集的信息到对应的txt文件中
    # for index,dataset in enumerate([train_dataset,eval_dataset]):
    #     file_name=''
    #     if index==0:
    #         file_name='train.txt'
    #     else:
    #         file_name='evl.txt'
    #     with open(file_name,'w') as out:
    #         for data in dataset:
    #             label=data[1]
    #             img_pair=data[2]
    #             img0=img_pair[0]
    #             img1=img_pair[1]
    #             out.write(f'{img0},{img1},{label}\n')



           
    
    # 定义模型以及模型初始化
    model=ResNet()
    model.blur.apply(kaiming_init_model)
    # model=ResNet()
    # model.blur.apply(kaiming_init_model)

    # 记录数据集相关信息
    args['logger'].info(f'train_dataset total length:{len(train_dataset)},eval_dataset total length:{len(eval_dataset) if eval_dataset else 0}, predict_dataset total length:{len(predict_dataset)}')
    args['logger'].info(f'train_dataset:{train_dataset.__class__.__name__}')
    args['logger'].info(f'model:{model.__class__.__name__}')

    # 构造训练对象
    trainer=Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        predict_dataset=predict_dataset
    )

    # 依据config中的mode进行训练、验证或者预测
    if args["mode"] == "train":
        trainer.train()
    elif args["mode"] == "eval":
        trainer.eval()
    elif args["mode"] == "predict":
        trainer.predict()
    else:
        args["logger"].info(f"Unrecognized trainer mode: {args['mode']}")

    progress.stop()

    if args["device"] != "CPU":
        destroy_process_group()

    args["logger"].info("Program Exited Normally")    