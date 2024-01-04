from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support,confusion_matrix


def calculate_metrics(logger, metrics_list, eval_result):
    eval_result = tuple(eval_result)
    metrics_list = metrics_list.split("_")
    metrics_dict = {}
    for name in metrics_list:
        if name == "Recall":
            metrics_dict["Recall"] = calculate_recall(eval_result,'None')
            metrics_dict["Recall_macro"] = calculate_recall(eval_result,'macro')
        elif name == "Precision":
            metrics_dict["Precision"] = calculate_precision(eval_result,'None')
            metrics_dict["Precision_macro"] = calculate_precision(eval_result,'macro')
        elif name == "F1score":
            metrics_dict["F1score"] = calculate_f1score(eval_result,'None')
            metrics_dict["F1score_macro"] = calculate_f1score(eval_result,'macro')
        else:
            logger.error(f"Please provide specific {name} functions")
    # metrics_dict["confusion_matrix"] = calculate_cfmatrix(eval_result)
    return metrics_dict

def calculate_cfmatrix(eval_result):
    label, prediction = eval_result
    return confusion_matrix(label, prediction)

def calculate_f1score(eval_result,average):
    # average为需要调节参数，平均值用macro，每个类别都求值用None
    if average=='None':
        average=None
    label, prediction = eval_result
    f1s=f1_score(label, prediction, average=average, zero_division=0)
    if average=='macro':
        return round(f1s,6)
    else:
        metrics_sub_dict={}
        for index,f1_single in enumerate(f1s):
            metrics_sub_dict[str(index-1)]=round(f1_single,6)
        return metrics_sub_dict


def calculate_precision(eval_result,average):
    # average为需要调节参数，平均值用macro，每个类别都求值用None
    if average=='None':
        average=None
    label, prediction = eval_result
    precisions=precision_score(label, prediction, average=average, zero_division=0)
    if average=='macro':
        return round(precisions,6)
    else:
        metrics_sub_dict={}
        for index,precision_single in enumerate(precisions):
            metrics_sub_dict[str(index-1)]=round(precision_single,6)
        return metrics_sub_dict
        # return round(
    #     precision_score(label, prediction, average=average, zero_division=0), 6)


def calculate_recall(eval_result,average):
    # average为需要调节参数，平均值用macro，每个类别都求值用None
    if average=='None':
        average=None
    label, prediction = eval_result
    # return round(recall_score(label, prediction, average="macro", zero_division=0), 6)
    recalls=recall_score(label, prediction, average=average, zero_division=0)
    if average=='macro':
        return round(recalls,6)
    else:
        metrics_sub_dict={}
        for index,recall_single in enumerate(recalls):
            metrics_sub_dict[str(index-1)]=round(recall_single,6)
        return metrics_sub_dict