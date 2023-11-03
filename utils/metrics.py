from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(logger, metrics_list, eval_result):
    eval_result = tuple(eval_result)
    metrics_list = metrics_list.split("_")
    metrics_dict = {}
    for name in metrics_list:
        if name == "F1score":
            metrics_dict["F1score"] = calculate_f1score(eval_result)
        elif name == "Precision":
            metrics_dict["Precision"] = calculate_precision(eval_result)
        elif name == "Recall":
            metrics_dict["Recall"] = calculate_recall(eval_result)
        else:
            logger.error(f"Please provide specific {name} functions")
    return metrics_dict


def calculate_f1score(eval_result):
    label, prediction = eval_result
    return round(f1_score(label, prediction, average="macro", zero_division=0), 6)


def calculate_precision(eval_result):
    label, prediction = eval_result
    return round(
        precision_score(label, prediction, average="macro", zero_division=0), 6
    )


def calculate_recall(eval_result):
    label, prediction = eval_result
    return round(recall_score(label, prediction, average="macro", zero_division=0), 6)