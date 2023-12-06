import argparse
import torch.distributed as dist
import torch
import os
from .set_logger import DDPLogger
from .set_seed import set_seed
import yaml

ENVIRONMENT_VARIABLES = ["LOCAL_RANK","RANK","GROUP_RANK","ROLE_RANK","LOCAL_WORLD_SIZE","WORLD_SIZE","ROLE_WORLD_SIZE","MASTER_ADDR","MASTER_PORT"]

def config_args():
    parser=argparse.ArgumentParser(description='my own pytoch')
    parser.add_argument('--gpu',type=str,required=True,help='use gpu')
    parser.add_argument('--config-file',type=str,required=True,help='config_file path')
    parser.add_argument('--datetime',type=str,required=True,help='datetime')

    args=parser.parse_args()
    args=get_other_config(args)

    return args

def get_other_config(args):
    

    # 获取当前配置的设备信息
    device,global_device=get_device(args)
    
    # 设置日志
    my_logger=DDPLogger(args.datetime,global_device,device).get_logger()
    
    logger_datetime=args.datetime

    # 查看环境变量
    if device!='CPU':
        check_ddp_var(my_logger)
    
    # 读取配置文件
    with open(args.config_file,'r') as f:
        args=yaml.load(f,Loader=yaml.FullLoader)

    # 设置随机种子
    set_seed(args['seed'])

    args['logger']=my_logger
    args['device']=device
    args['global_device']=global_device
    args['world_size']=int(os.environ['WORLD_SIZE'])
    args['datetime']=logger_datetime

    return args

def get_device(args):
    gpu_list = [int(i) for i in args.gpu.split(',')]
    if gpu_list[0]!=-1:

        ddp_setup()
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
        device = "GPU:{}".format(gpu_list[int(os.environ["LOCAL_RANK"])])
        global_device = "GPU:{}".format(gpu_list[0])
    else:
        device='CPU'
        global_device='CPU'
    return device,global_device        

def ddp_setup():    
    # 以nccl后端方式建立进程间通信
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def check_ddp_var(my_logger):
    for var in ENVIRONMENT_VARIABLES:
        my_logger.info('{}:{}'.format(var,os.environ[var]))



    

