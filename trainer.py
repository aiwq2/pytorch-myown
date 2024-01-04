import os
import torch
import torch.nn as nn
from torch.optim import AdamW,SGD
from utils.rich_tqdm import progress
from utils.metrics import calculate_metrics
from utils.write_tensorboard import write_tensorboard
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_gather_object, barrier
from transformers import get_scheduler
from datetime import datetime as dt
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
                    # self.best_metric=self.snapshot['BEST_METRIC']
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
                    # self.best_metric=self.snapshot['BEST_METRIC']
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
        if self.args['use_tensorboard']:
            self.tensorboard_writer=SummaryWriter(
                os.path.join('tensorboard_result',dt.strftime(dt.strptime(args['datetime'],'%Y_%m_%d_%H_%M_%S'),'%Y_%m_%d'),args['device'],args['datetime'])
            )

        self.early_stop_sign=deque(maxlen=args['early_stop']['patience'])            

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
        
        self.logger.info(f"Resuming from {self.args['model_path']['snapshot']} at Epoch {snapshot['EPOCHS_RUN']}")
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
        # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
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
        for epoch in range(0,self.args['max_epochs']+1):

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
            metrics_train=None

            # # 评估模型在训练集上的效果
            # metrics_train,train_loss=self.eval_one_epoch(
            #     dataloader=self.train_eval_dataloader,task_id=rich_train_eval_step_id,epoch=epoch
            # )
            # self.logger.info(f"Train Eval Epoch: {epoch} Metrics: {metrics_train}")

            # 评估模型在验证集上面的效果
            metrics_eval=None
            if self.eval_dataset:
                metrics_eval,evl_loss = self.eval_one_epoch(
                    dataloader=self.eval_dataloader, task_id=rich_eval_step_id,epoch=epoch
                )
                self.logger.info(f'learning rate:{self.optimizer.param_groups[0]["lr"]}')
                self.logger.info(f"Eval Epoch: {epoch} Loss: {evl_loss}")
                self.logger.info(f"Eval Epoch: {epoch} Metrics: {metrics_eval}")

            barrier()

            if self.args['use_tensorboard']:
                write_tensorboard(self.tensorboard_writer,epoch,loss,evl_loss,metrics_train,metrics_eval,self.args['metric_average'])
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

    def train_one_epoch(self,dataloader,task_id,epoch=0):
        self.model.train()
        progress.reset(task_id)

        loss_list=[]

        if self.args['device']!='CPU':
            dataloader.sampler.set_epoch(epoch)

        for _,train_data in enumerate(dataloader):
            loss_single=self.one_batch(train_data,'train',epoch=epoch)
            loss_list.append(loss_single)
            progress.update(task_id=task_id,advance=1,loss=loss_single)
        
        return sum(loss_list)/len(loss_list)
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
        metrics,train_loss = self.eval_one_epoch(
            dataloader=self.train_eval_dataloader, task_id=rich_train_eval_step_id
        )
        self.logger.info(f"Train Eval loss: {train_loss}")
        self.logger.info(f"Train Eval Metrics: {metrics}")
        # 评估模型在验证集上面的效果
        metrics,eval_loss = self.eval_one_epoch(
            dataloader=self.eval_dataloader, task_id=rich_eval_step_id
        )
        self.logger.info(f"Eval loss: {eval_loss}")
        self.logger.info(f"Eval Metrics: {metrics}")
    
    def eval_one_epoch(self,task_id,dataloader,epoch=0):

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
        self.all_loss_list=[[] for i in range(self.args['world_size'])]
        temp_loss_result=[]
        for _,eval_data in enumerate(dataloader):
            eval_result,loss=self.one_batch(eval_data,'eval',epoch=epoch)

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
            temp_loss_result.append(loss)
            all_gather_object(self.all_loss_list,temp_loss_result)
            

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
                self.all_eval_result
            ),sum(self.all_loss_list[0])/len(self.all_loss_list[0])
        else:
            return {'Not Main Process':0.0}
    # 预测主函数
    def predict(self):
        self.predict_dataloader = self.prepare_dataloader(
            self.predict_dataset, self.args["batch_size"]["predict"]
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
        # self.logger.info(result)
        file_save='online_data_test_result_iteration13.txt'
        with open(file_save,'w') as out:
            for label,img in result:
                out.write(f'{label.item():.2f},{img}\n')
        self.logger.info(f"Predict Result Saved at {file_save}")
        self.logger.info("Predict Finished")

    def predict_one_epoch(self,task_id,dataloader):
        self.model.eval()

        progress.reset(task_id)

        result_list=[]
        for _,eval_data in enumerate(dataloader):
            result_single,imgs=self.one_batch(eval_data,'predict')
            combined_list=[(a, b) for a, b in zip(result_single, imgs)]
            result_list.extend(combined_list)
            progress.update(task_id,advance=1)

        return result_list
    
    def one_batch(self,data,mode,epoch=0):
        data_need_to_cuda, labels,contents = data
       
        if isinstance(data_need_to_cuda,list):
            for data_name_index in range(len(data_need_to_cuda)):
                if torch.is_tensor(data_need_to_cuda[data_name_index]):
                    data_need_to_cuda[data_name_index] = data_need_to_cuda[data_name_index].to(
                        self.gpu_id, dtype=torch.float
                    )
            labels = labels.to(self.gpu_id)
            # feature1=feature1.to(self.gpu_id)
            # feature2=feature2.to(self.gpu_id)
        else:
            data_need_to_cuda=data_need_to_cuda.to(self.gpu_id)
            labels=labels.to(self.gpu_id)
            feature1=feature1.to(self.gpu_id)
            feature2=feature2.to(self.gpu_id)
            
        if mode=='train':
            self.optimizer.zero_grad()
            loss_single=self.model(epoch,mode,self.criterion,labels,contents,self.args['extra']['delta'],data_need_to_cuda[0],data_need_to_cuda[1])


            loss_single.backward()
            self.optimizer.step()
            self.scheduler.step(loss_single)
            return loss_single.item()
        
        elif mode=='eval':
            with torch.no_grad():
                return self.model(epoch,mode,self.criterion,labels,contents,self.args['extra']['delta'],data_need_to_cuda[0],data_need_to_cuda[1])
        elif mode=='predict':
            with torch.no_grad():
                return self.model(epoch,mode,self.criterion,labels,contents,self.args['extra']['delta'],data_need_to_cuda[0]),data_need_to_cuda[1]
        
        

    def judge_whether_save_and_stop(self,epoch,metrics_eval):
        main_metric_name=self.args["main_metric"]+'_macro'
        if  metrics_eval[main_metric_name] > self.best_metric+self.args['early_stop']['min_delta']:
            if metrics_eval[main_metric_name] > self.best_metric+self.args['early_stop']['min_delta']:
                self.logger.info(
                    f"Best Metrics Now: {self.args['main_metric']}:{metrics_eval[main_metric_name]} > Best Metrics Before:{self.args['main_metric']}: {self.best_metric}"
                )
                self.best_metric = metrics_eval[main_metric_name]
                self.early_stop_sign.append(0)
            else:
                self.early_stop_sign.append(1)
            if self.args["model_path"]["save"]:
                self.save_snapshot(epoch,metrics_eval[main_metric_name])
        
            
        else:
            self.early_stop_sign.append(1)
            # 早停，无法进行进程间的同步，由于是deque，所以得连续patience个0，才会停止
            if sum(self.early_stop_sign)==self.args['early_stop']['patience']:
                self.logger.info(
                    f'The Effect of last {self.args["early_stop"]["patience"]} epochs has not improved! Early Stop!'
                )
                self.logger.info(f"Best Metric:{self.args['main_metric']}:{self.best_metric}")
                return True
        return False

    def save_snapshot(self,epoch,best_metric):
        snapshot={
            # 注意是model.module，因为使用了DDP
            'MODEL_STATE':self.model.module.state_dict(),
            'EPOCHS_RUN':epoch,
            'OPTIMIZER_STATE':self.optimizer.state_dict(),
            'SCHEDULER_STATE':self.scheduler.state_dict(),
            'BEST_METRIC':best_metric
        }
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)
        torch.save(
            snapshot,
            os.path.join(self.snapshot_path,self.args['model_path']['snapshot'])
        )
        self.logger.info(
            f"Epoch {epoch} Training snapshot saved at snapshot_path/{self.args['model_path']['snapshot']}"
        )