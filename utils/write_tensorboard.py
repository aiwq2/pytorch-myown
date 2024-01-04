def write_tensorboard(tensorboard_writer,epoch, loss,evl_loss, metrics_train, metrics_eval,metric_average):
    tensorboard_writer.add_scalars(f'Loss',{'train_loss':loss,'evl_loss':evl_loss},epoch)
    # 为None则绘制对应标签的图像
    # if metric_average=='None':
    #     for metric in list(metrics_train.keys()):
    #         labels=[label for label in list(metrics_train[metric].keys())]
    #         scalars_dict={}
    #         for label in labels:
    #             if metrics_eval:
    #                 scalars_dict[f'_{label}_Train']=metrics_train[metric][label]
    #                 scalars_dict[f'_{label}_Eval']=metrics_eval[metric][label]
    #             else:
    #                 scalars_dict[f'_{label}_Train']=metrics_train[metric][label]
    #         if metrics_eval:
    #             tensorboard_writer.add_scalars(
    #                 f'{metric}',
    #                 scalars_dict,
    #                 epoch
    #             )
    #         else:
    #             tensorboard_writer.add_scalars(
    #                 f'{metric}',
    #                 scalars_dict,
    #                 epoch
    #             )
    # # 否则绘制metric平均值的图像
    # else:
    #     for metric in list(metrics_train.keys()):
    #         if isinstance(metrics_train[metric],(int,float)):
    #             if metrics_eval:
    #                 tensorboard_writer.add_scalars(
    #                     f'{metric}',
    #                     {f'_Train':metrics_train[metric],f'_Eval':metrics_eval[metric]},
    #                     epoch
    #                 )
    #             else:
    #                 tensorboard_writer.add_scalar(
    #                     f'{metric}',
    #                     metrics_train[metric],
    #                     epoch
    #                 )