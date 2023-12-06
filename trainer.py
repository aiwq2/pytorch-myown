import os
import torch
import torch.nn as nn
from torch.optim import AdamW,SGD
from utils.rich_tqdm import progress
from utils.metrics import calculate_metrics
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_gather_object, barrier
from transformers import get_scheduler

from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union
from collections import deque


class Trainer:
    def __init__(
            self,
            args=None,
            model=None,
            train_dataset:Dataset=None,
            eval_dataset:Optional[Dataset] =None,
            predict_dataset:Optional[Dataset] =None
    ):
        self.args=args
        self.logger=args['logger']
        self.mode=args['mode']

        # 当前模型的最好评价指标
        self.best_metric=0
        # 起始epoch
        self.epoch_start=1

        # snapshot
        # self.snapshot_path=os.path.join(
        #     self.args['model_path']['snapshot_dir'],self.args['datetime']
        # )
        self.snapshot_path=self.args['model_path']['snapshot_dir']
        self.snapshot=None


        # 校验传递的模型
        if not model:
            self.logger.error("Trainer requires a model argument")
            return

        if self.args['device']!='CPU':
            self.gpu_id=int(os.environ['LOCAL_RANK'])
            
            if self.args['get_from_snapshot']:
                self.snapshot=self.load_snapshot()

                if self.snapshot is not None:
                    self.epoch_start = self.snapshot["EPOCHS_RUN"] + 1
                    model.load_state_dict(self.snapshot["MODEL_STATE"])
                else:
                    self.logger.error('we need snapshot path,but we did not get')
                    raise ValueError('snapshot path is not given')
            
            self.model = DistributedDataParallel(
                model.to(self.gpu_id), find_unused_parameters=False
            )
        else:
            self.gpu_id=None
            if self.args['get_from_snapshot']:
                self.snapshot=self.load_snapshot()
                if self.snapshot:
                    self.epoch_start=self.snapshot('EPOCHS_RUN')+1
                    model.load_state_dict(self.snapshot["MODEL_STATE"])
                else:
                    self.logger.error('we need snapshot path,but we did not get')
                    raise ValueError('snapshot path is not given')

            self.model = model

        # 检验数据集是否存在                
        if train_dataset is None and self.mode == "train":
            self.logger.warning("No Train Dataset Passed")
        self.train_dataset = train_dataset

        # 校验验证数据集
        if eval_dataset is None and self.mode != "predict":
            self.logger.warning("No Eval Dataset Passed")
        self.eval_dataset = eval_dataset

        # 校验测试数据集
        if predict_dataset is None and self.mode == "predict":
            self.logger.warning("No Predict Dataset Passed")
        self.predict_dataset = predict_dataset    

        # 优化器与损失函数
        self.criterion=self.get_criterion()    
        self.optimizer,self.scheduler=self.get_optimizer_and_scheduler()

        if self.snapshot:
            self.optimizer.load_state_dict(self.snapshot["OPTIMIZER_STATE"])
            self.scheduler.load_state_dict(self.snapshot["SCHEDULER_STATE"])

        self.tensorboard_writer=SummaryWriter(
            os.path.join('tensorboard_result',args['datetime'],args['device'])
        )

        self.early_stop_sign=deque(maxlen=args['early_stop'])            

    def load_snapshot(self):
        if not os.path.exists(
            os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"])
        ):
            self.logger.warning(f"{self.snapshot_path} not exist")
            return None
        if self.gpu_id != None:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(
                os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"]),
                map_location=loc,
            )
        else:
            snapshot = torch.load(
                os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"])
            )
        
        self.logger.info(f"Resuming from snapshot at Epoch {snapshot['EPOCHS_RUN']}")
        return snapshot
    
    def get_criterion(self):
        if self.args["criterion"] == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.args["criterion"] == "DualLoss":
            from loss.DualLoss import DualLoss
            return DualLoss(0.5,0.1)
        elif self.args['criterion']=='FocalLoss':
            from loss.FocalLoss import FocalLoss
            return FocalLoss()
        elif self.args['criterion']=='PairwiseLoss':
            from loss.PairwiseLoss import PairwiseLoss
            return PairwiseLoss()
        else:
            self.args.error('we cannot get criterion:{} you need,pls checkout!'.format(self.args['criterion']))
            raise ValueError('cannot get criterion')
        
    # 获取模型优化器和调度器
    def get_optimizer_and_scheduler(self):
        if self.args['optimizer']=='AdamW':
            optimizer=self.modified_optimizer()
        elif self.args['optimizer']=='SGD':
            optimizer=SGD(self.model.parameters(),lr=self.args['lr'],weight_decay=self.args['wd'],momentum=self.args['momentum'])
        else:
            self.logger.error('Unrecoginized Optimizer')
            exit()
        num_training_all_steps=(
            len(self.train_dataset)//(self.args['batch_size']['train'])*self.args['world_size']
        )*self.args['max_epochs']
        self.logger.info(f'Total train steps:{num_training_all_steps}')
        scheduler=get_scheduler(
            self.args['scheduler']['name'],
            optimizer,
            num_training_all_steps*self.args['scheduler']['ratio'],
            num_training_all_steps
        )
        return optimizer,scheduler

    def modified_optimizer(self):
        param_optimizer=list(self.model.named_parameters())
        no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
        optimizer_grouped_parameters=[
            {
                'params':[
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':self.args['wd'],
            },
            {
                'params':[
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                'weight_decay':0.0
            }


        ]
        return AdamW(optimizer_grouped_parameters,lr=self.args['lr'])

    def prepare_dataloader(self,dataset,batch_size):
        if self.gpu_id>=0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                sampler=DistributedSampler(dataset=dataset,shuffle=True,seed=self.args['seed'])
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )

    # 训练主函数
    def train(self):

        # 计算epoch进度条
        rich_epoch_id=progress.add_task(
            description='Epoch',total=self.args["max_epochs"]
        )
        
        # 这里只判断了训练集，我觉得可能存在不使用验证的地方
        if  not self.train_dataset:
            self.logger.error("No Train or Eval Dataset Passed")
            return

        self.train_dataloader= self.prepare_dataloader(
            self.train_dataset,self.args['batch_size']['train']
        )

        rich_train_step_id = progress.add_task(
            description="Train Step", total=len(self.train_dataloader)
        )

        # 相当于用训练集来进行了验证
        self.train_eval_dataloader = self.prepare_dataloader(
            self.train_dataset, self.args["batch_size"]["eval"]
        )
        rich_train_eval_step_id = progress.add_task(
            description="Train Eval Step", total=len(self.train_eval_dataloader)
        )            

        # 如果存在验证集则载入验证集
        if self.eval_dataset:
            self.eval_dataloader = self.prepare_dataloader(
                self.eval_dataset, self.args["batch_size"]["eval"]
            )
            rich_eval_step_id = progress.add_task(
                description="Eval Step", total=len(self.eval_dataloader)
            )
        
        # 启动终端进度条
        if self.args["device"] == "CPU" or (self.args["device"] != "CPU" and self.args["device"] == self.args["global_device"]
        ):
            progress.start() 

        # 开始循环训练   
        self.logger.info('begin to train')        
        for epoch in range(1,self.args['max_epochs']+1):

            self.logger.trace('-'*20)
            if epoch<self.epoch_start:
                progress.update(rich_epoch_id,advance=1)
                continue

            loss=self.train_one_epoch(
                dataloader=self.train_dataloader,
                task_id=rich_train_step_id,
                epoch=epoch
            )

            self.logger.info(f'Train Epoch:{epoch} Loss:{loss}')
            torch.cuda.empty_cache()

            # 评估模型在训练集上的效果
            metrics_train=self.eval_one_epoch(
                dataloader=self.train_eval_dataloader,task_id=rich_train_eval_step_id
            )
            self.logger.info(f"Train Eval Epoch: {epoch} Metrics: {metrics_train}")
            # 评估模型在验证集上面的效果
            metrics_eval=None
            if self.eval_dataset:
                metrics_eval = self.eval_one_epoch(
                    dataloader=self.eval_dataloader, task_id=rich_eval_step_id
                )
                self.logger.info(f"Eval Epoch: {epoch} Metrics: {metrics_eval}")

            barrier()

            self.write_tensorboard(epoch,loss,metrics_train,metrics_eval)
            if self.args['device']=='CPU' or (
                self.args["device"] != "CPU"
                and self.args["device"] == self.args["global_device"]
            ):
                if self.eval_dataset:
                    if self.judge_whether_save_and_stop(epoch,metrics_eval):
                        return
                
            # 更新进度条
            progress.update(rich_epoch_id, advance=1)

        self.logger.info(f"Best Metric: {self.best_metric}")


    # 评估主函数
    def eval(self):
        self.train_eval_dataloader = self.prepare_dataloader(
            self.train_dataset, self.args["batch_size"]["eval"]
        )
        rich_train_eval_step_id = progress.add_task(
            description="Train Eval Step", total=len(self.train_eval_dataloader)
        )

        self.eval_dataloader = self.prepare_dataloader(
            self.eval_dataset, self.args["batch_size"]["eval"]
        )
        rich_eval_step_id = progress.add_task(
            description="Eval Step", total=len(self.eval_dataloader)
        )
        if self.args["device"] == "CPU" or (
            self.args["device"] != "CPU"
            and self.args["device"] == self.args["global_device"]
        ):
            progress.start()

        # 评估模型在训练集上面的效果
        metrics = self.eval_one_epoch(
            dataloader=self.train_eval_dataloader, task_id=rich_train_eval_step_id
        )
        self.logger.info(f"Train Eval Metrics: {metrics}")
        # 评估模型在验证集上面的效果
        metrics = self.eval_one_epoch(
            dataloader=self.eval_dataloader, task_id=rich_eval_step_id
        )
        self.logger.info(f"Eval Metrics: {metrics}")

    # 预测主函数
    def predict(self):
        self.predict_dataloader = self.prepare_dataloader(
            self.predict_dataset, self.args["batch_size"]["eval"]
        )
        rich_predict_step_id = progress.add_task(
            description="Predict Step", total=len(self.predict_dataloader)
        )

        # 启动终端进度条
        if self.args["device"] == "CPU" or (
            self.args["device"] != "CPU"
            and self.args["device"] == self.args["global_device"]
        ):
            progress.start()

        # 给测试集打标签
        result = self.predict_one_epoch(
            dataloader=self.predict_dataloader, task_id=rich_predict_step_id
        )
        self.logger.info(result)

        self.logger.info("Predict Finished")

    def train_one_epoch(self,dataloader,task_id,epoch):
        self.model.train()
        progress.reset(task_id)

        loss_list=[]

        if self.args['device']!='CPU':
            dataloader.sampler.set_epoch(epoch)

        for _,train_data in enumerate(dataloader):
            loss_single=self.one_batch(train_data,'train')
            loss_list.append(loss_single)
            progress.update(task_id=task_id,advance=1,loss=loss_single)
        
        return sum(loss_list)/len(loss_list)
    
    def eval_one_epoch(self,task_id,dataloader):

            #         gathered_image_features = [
            #     torch.zeros_like(image_features) for _ in range(world_size)
            # ]
            # gathered_text_features = [
            #     torch.zeros_like(text_features) for _ in range(world_size)
            # ]
            
            # dist.all_gather(gathered_image_features, image_features)
            # dist.all_gather(gathered_text_features, text_features)

            # all_image_features = torch.cat(
            #     [image_features]
            #     + gathered_image_features[:rank]
            #     + gathered_image_features[rank + 1 :]
            # )
            # all_text_features = torch.cat(
            #     [text_features]
            #     + gathered_text_features[:rank]
            #     + gathered_text_features[rank + 1 :]
            # )

        self.model.eval()

        progress.reset(task_id)

        self.all_eval_result=[]
        temp_eval_result=[]

        for _,eval_data in enumerate(dataloader):
            eval_result=self.one_batch(eval_data,'eval')

            # len(eval_result)是model返回的元组的长度，这里为2，一个是label，一个是预测值
            while len(self.all_eval_result)!=len(eval_result):
                self.all_eval_result.append(
                    [[] for i in range(self.args['world_size'])]
                )
            while len(temp_eval_result)!=len(eval_result):
                temp_eval_result.append([])
            for i,eval_single_result in enumerate(eval_result):
                temp_eval_result[i].extend(eval_single_result)
                all_gather_object(self.all_eval_result[i],temp_eval_result[i])

            progress.update(task_id=task_id,advance=1)
        
        if self.args["device"] == "CPU" or (
            self.args["device"] != "CPU"
            and self.args["device"] == self.args["global_device"]
        ):
            for i, result in enumerate(self.all_eval_result):
                merge_list = []
                for rank_list in result:
                    merge_list = merge_list + rank_list
                self.all_eval_result[i] = merge_list
            return calculate_metrics(
                self.logger,
                self.args['metrics'],
                self.all_eval_result,
            )
        else:
            return {'Not Main Process':0.0}
        
    def predict_one_epoch(self,task_id,dataloader):
        self.model.eval()

        progress.reset(task_id)

        result_list=[]
        for _,eval_data in enumerate(dataloader):
            _,result_single=self.one_batch(eval_data,'eval')
            result_list.extend(result_single)
            progress.update(task_id,advance=1)

        return result_list
            
    

    def one_batch(self,data,mode):
        data_need_to_cuda, labels = data
        if isinstance(data_need_to_cuda,list):
            for data_name in data_need_to_cuda:
                data_need_to_cuda[data_name] = data_need_to_cuda[data_name].to(
                    self.gpu_id, dtype=torch.long
                )
        else:
            data_need_to_cuda=data_need_to_cuda.to(self.gpu_id)
            labels=labels.to(self.gpu_id)
        
        if mode=='train':
            self.optimizer.zero_grad()
            loss_single=self.model(mode,self.criterion,data_need_to_cuda,labels,self.args['extra']['delta'])


            loss_single.backward()
            self.optimizer.step()
            self.scheduler.step()
            return loss_single.item()
        
        elif mode=='eval':
            with torch.no_grad():
                return self.model(mode,self.criterion,data_need_to_cuda,labels,self.args['extra']['delta'])
            
        
    def write_tensorboard(self,epoch, loss, metrics_train, metrics_eval):
        self.tensorboard_writer.add_scalar(f'Loss',loss,epoch)
        for metric in list(metrics_train.keys()):
            if metrics_eval:
                self.tensorboard_writer.add_scalars(
                    f'{metric}/',
                    {'Train_'+metric:metrics_train[metric],'Eval_'+metric:metrics_eval[metric]},
                    epoch+1
                )
            else:
                self.tensorboard_writer.add_scalar(
                    f'{metric}',
                    metrics_train[metric],
                    epoch+1
                )

    def judge_whether_save_and_stop(self,epoch,metrics):
        if metrics[self.args["main_metric"]] > self.best_metric:
            self.logger.info(
                f"Best Metrics Now: {self.args['main_metric']}:{metrics[self.args['main_metric']]} > Best Metrics Before:{self.args['main_metric']}: {self.best_metric}"
            )
            self.best_metric = metrics[self.args["main_metric"]]
            if self.args["model_path"]["save"]:
                self.save_snapshot(epoch)
            self.early_stop_sign.append(0)
        else:
            self.early_stop_sign.append(1)
            # 早停，无法进行进程间的同步
            if sum(self.early_stop_sign)==self.args['early_stop']:
                self.logger.info(
                    f'The Effect of last {self.args["early_stop"]} epochs has not improved! Early Stop!'
                )
                self.logger.info(f"Best Metric:{self.args['main_metric']}:{self.best_metric}")
                return True
        return False

    def save_snapshot(self,epoch):
        snapshot={
            'MODEL_STATE':self.model.module.state_dict(),
            'EPOCHS_RUN':epoch,
            'OPTIMIZER_STATE':self.optimizer.state_dict(),
            'SCHEDULER_STATE':self.scheduler.state_dict(),
        }
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot)
        torch.save(
            snapshot,
            os.path.join(self.snapshot_path,self.args['model_path']['snapshot']),
        )
        self.logger.info(
            f"Epoch {epoch} Training snapshot saved at {self.snapshot_path}"
        )