from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def calculate_metrics(logger, eval_result):
    eval_result = tuple(eval_result)
    metrics_dict = {}
    metrics_dict=get_metrics_dict(logger,eval_result,metrics_dict)
    # 直接打印混淆矩阵
    # metrics_dict["confusion_matrix"] = calculate_cfmatrix(eval_result)
    # print(calculate_cfmatrix(eval_result))
    return metrics_dict

# 混淆矩阵
def calculate_cfmatrix(eval_result):
    label, prediction,thresholds = eval_result
    return confusion_matrix(label, prediction)

# PR曲线图
def get_precision_recall_curve(logger,label,logits):
    # 计算Precision-Recall Curve中的precision、recall和阈值

    # AUC的thresholds取决于drop_intermediate参数，如果为False，那么就是取所有样本score，如果为True，则是取部分间隔
    # score，以使得整个图像显得平滑，threholds取负样本的score而PR曲线图的thresholds取所有样本的score
    precision, recall, thresholds = precision_recall_curve(label, logits)

    # 计算平均精度（Average Precision）
    avg_precision = average_precision_score(label, logits)

    # 绘制PR Curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP={avg_precision:.2f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc='best')
    # save_file='PR_curve.jpg'
    # plt.savefig(save_file)
    # logger.info(f'{save_file} saved')
    return round(avg_precision,2)

# ROC曲线图
def get_ROC(logger,labels,logits):
    # 计算ROC曲线
    
    # AUC的thresholds取决于drop_intermediate参数，如果为False，那么就是取所有样本score，如果为True，则是取部分间隔
    # score，以使得整个图像显得平滑，threholds取负样本的score而PR曲线图的thresholds取所有样本的score
    fpr, tpr, thresholds = roc_curve(labels, logits,drop_intermediate=False)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    

    # 绘制ROC曲线
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC={roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='best')
    # save_file='ROC_curve.jpg'
    # plt.savefig(save_file)
    # logger.info(f'{save_file} saved')
    return round(roc_auc,2)

def get_metrics_dict(logger,eval_result,metrics_dict):
    label, prediction,logits = eval_result
    label_set=get_label(label)

    # PR曲线图
    ap=get_precision_recall_curve(logger,label,logits)
    # ROC曲线图
    roc_auc=get_ROC(logger,label,logits)

    # f1_score
    f1s_individual=f1_score(label, prediction, average=None, zero_division=0)
    f1s_macro=f1_score(label, prediction, average='macro', zero_division=0)

    # precision_score
    precision_individual=precision_score(label, prediction, average=None, zero_division=0)
    precision_macro=precision_score(label, prediction, average='macro', zero_division=0)

    # recall_score
    recall_individual=recall_score(label, prediction, average=None, zero_division=0)
    recall_macro=recall_score(label, prediction, average='macro', zero_division=0)

    metrics_dict['Recall_individual']={}
    metrics_dict['Recall_macro']=round(recall_macro,6)  
    metrics_dict['Precision_individual']={}
    metrics_dict['Precision_macro']=round(precision_macro,6)
    metrics_dict['F1score_individual']={}
    metrics_dict['F1score_macro']=round(f1s_macro,6)
    metrics_dict['AP']=ap
    metrics_dict['ROC_AUC']=roc_auc

    for index,f1_single in enumerate(f1s_individual):
        metrics_dict['F1score_individual'][label_set[index]]=round(f1_single,6)
    for index,precison_single in enumerate(precision_individual):
        metrics_dict['Precision_individual'][label_set[index]]=round(precison_single,6)
    for index,recall_single in enumerate(recall_individual):
        metrics_dict['Recall_individual'][label_set[index]]=round(recall_single,6)
    return metrics_dict

def get_label(labels):
    label_set=set()
    for lb in labels:
        label_set.add(lb)
    label_set=list(label_set)
    label_set.sort()
    return label_set