from utils.params import config_args
import os
from loguru import logger
from utils.rich_tqdm import progress
from trainer import Trainer
from utils.get_bert_and_tokenizer import getBert
from torch.distributed import destroy_process_group
from dataset.MNIST import MNIST_Test_data,MNIST_Train_data
from dataset.BlUR.blur_compare import BlurPair
from model.CNN import MyCNN,init_MyCNN
from model.Resnet import ResNet
from model.model_init import init_model
from utils.dataset_split import dataset_split_sklearn,dataset_split_torch








if __name__=='__main__':

    # print(os.environ['MASTER_ADDR'])
    # print(os.environ['MASTER_PORT'])
    # print(os.environ['RANK'])
    # print(os.environ['LOCAL_RANK'])
    # print(os.environ['WORLD_SIZE'])

    args=config_args()
    for k,v in sorted(args.items(),key=lambda x:x[0]):
        args['logger'].info(f'{k}:{v}')

    # 定义dataset
    train_dataset=BlurPair()
    eval_dataset=[]
    predict_dataset=[]

    # 划分训练集和验证集
    if args['split_dataset']:
        train_dataset,eval_dataset=dataset_split_torch(train_dataset,test_size=args['split_test_ratio'])
           
    args['logger'].info(f'train_dataset total length:{len(train_dataset)},eval_dataset total length:{len(eval_dataset) if eval_dataset else 0}, predict_dataset total length:{len(predict_dataset)}')

    # 定义
    model=ResNet()
    model.blur.apply(init_model)



    trainer=Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        predict_dataset=predict_dataset
    )

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