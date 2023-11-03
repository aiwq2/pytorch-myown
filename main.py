from utils.params import config_args
import os
from loguru import logger
from utils.rich_tqdm import progress
from dataset.StanceDataset_VAST import StanceDataset_VAST
from dataset.StanceDataset_VAST_simple import StanceDataset_VAST_simple
from dataset.StanceDataset_Bart_VAST import StanceData
from dataset.LineByLineTextDataset import LineByLineTextDataset
from trainer import Trainer
from utils.get_bert_and_tokenizer import getBert
from torch.distributed import destroy_process_group














if __name__=='__main__':

    print(os.environ['MASTER_ADDR'])
    print(os.environ['MASTER_PORT'])
    print(os.environ['RANK'])
    print(os.environ['LOCAL_RANK'])
    print(os.environ['WORLD_SIZE'])

    args=config_args()
    args['logger'].info('args info:{}'.format(args))

    
    
    # 引入模型
    # max_length=512
    # train_dataset = LineByLineTextDataset(
    #     args=args,
    #     file_path=os.path.join(
    #         args["data_path"]["data_dir"], args["data_path"]["train"]
    #     ),
    #     max_length=max_length
    # )

    

    trainer=Trainer(
        args=args,
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