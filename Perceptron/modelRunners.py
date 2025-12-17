def model_runner_for_each_data_percentage(data, label, lr, epoch, model_function):
    splits = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    model_train_accuracy = []
    model_test_accuracy = []
    model_train_precision = []
    model_test_precision = []
    model_train_recall = []
    model_test_recall = []
    model_train_f1 = []
    model_test_f1 = []
    for i in splits:
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
            train_f1_measure,
            test_f1_measure,
        ) = model_function(data, label, lr, epoch, i)
        model_train_accuracy.append(train_acc)
        model_train_precision.append(train_precision)
        model_train_recall.append(train_recall)
        model_train_f1.append(train_f1_measure)
        model_test_accuracy.append(test_acc)
        model_test_precision.append(test_precision)
        model_test_recall.append(test_recall)
        model_test_f1.append(test_f1_measure)
    return (
        model_train_accuracy,
        model_train_precision,
        model_train_recall,
        model_train_f1,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        model_test_f1,
        splits,
    )


def model_runner_for_each_learning_rate(data, label, epoch, model_function):
    lr = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
    ]
    model_train_accuracy = []
    model_test_accuracy = []
    model_train_precision = []
    model_test_precision = []
    model_train_recall = []
    model_test_recall = []
    model_train_f1 = []
    model_test_f1 = []
    for i in lr:
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
            train_f1_measure,
            test_f1_measure,
        ) = model_function(data, label, i, epoch, 1.0)
        model_train_accuracy.append(train_acc)
        model_train_precision.append(train_precision)
        model_train_recall.append(train_recall)
        model_train_f1.append(train_f1_measure)
        model_test_accuracy.append(test_acc)
        model_test_precision.append(test_precision)
        model_test_recall.append(test_recall)
        model_test_f1.append(test_f1_measure)
    return (
        model_train_accuracy,
        model_train_precision,
        model_train_recall,
        model_train_f1,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        model_test_f1,
        lr,
    )
