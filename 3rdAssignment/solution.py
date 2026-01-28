from runner import DecisionTree_metrics, LDA_metrics, SVM_metrics,comparison


def SVM_Scenario(weighted_input, label):
    kernels = ["linear", "rbf"]
    c = [1, 2, 3, 4, 5,6,7,8,9,10]
    filenames = [
        f"SVM({kernels[0]}, c={c[0]})",
        f"SVM({kernels[0]}, c={c[1]})",
        f"SVM({kernels[0]}, c={c[2]})",
        f"SVM({kernels[0]}, c={c[3]})",
        f"SVM({kernels[0]}, c={c[4]})",
        f"SVM({kernels[0]}, c={c[5]})",
        f"SVM({kernels[0]}, c={c[6]})",
        f"SVM({kernels[0]}, c={c[7]})",
        f"SVM({kernels[0]}, c={c[8]})",
        f"SVM({kernels[0]}, c={c[9]})",
        f"SVM({kernels[1]}, c={c[0]})",
        f"SVM({kernels[1]}, c={c[1]})",
        f"SVM({kernels[1]}, c={c[2]})",
        f"SVM({kernels[1]}, c={c[3]})",
        f"SVM({kernels[1]}, c={c[4]})",
        f"SVM({kernels[1]}, c={c[5]})",
        f"SVM({kernels[1]}, c={c[6]})",
        f"SVM({kernels[1]}, c={c[7]})",
        f"SVM({kernels[1]}, c={c[8]})",
        f"SVM({kernels[1]}, c={c[9]})",
    ]
    counter = 0

    for k in range(len(kernels)):
        for i in range(len(c)):
            SVM_metrics(
                weightedInput=weighted_input,
                label=label,
                kernel=kernels[k],
                C=c[i],
                fileName=filenames[counter],
            )
            counter += 1


def LDA_Scenario(weighted_input, label):
    solvers = ["svd", "lsqr" ]
    filenames = [
        f"LDA({solvers[0]})",
        f"LDA({solvers[1]})",
    ]
    counter = 0

    for k in range(len(solvers)):
        LDA_metrics(
            weightedInput=weighted_input,
            label=label,
            solver=solvers[k],
            fileName=filenames[counter],
        )
        counter += 1


def DecisionTree_Scenario(weighted_input, label):
    criterion = ["entropy", "gini", "log_loss"]
    max_depth = [4,5 ,10,20 ]
    min_split = [5, 5,4, 2]
    min_sample_leaf = [10, 5, 10, 20]
    filenames = [
        f"Decision Tree({criterion[0]} max_depth={max_depth[0]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[0]} max_depth={max_depth[1]},min_split={min_split[1]},min_sample_leaf={min_sample_leaf[1]})",
        f"Decision Tree({criterion[0]} max_depth={max_depth[2]},min_split={min_split[2]},min_sample_leaf={min_sample_leaf[2]})",
        f"Decision Tree({criterion[0]} max_depth={max_depth[3]},min_split={min_split[3]},min_sample_leaf={min_sample_leaf[3]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[0]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[1]},min_split={min_split[1]},min_sample_leaf={min_sample_leaf[1]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[2]},min_split={min_split[2]},min_sample_leaf={min_sample_leaf[2]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[3]},min_split={min_split[3]},min_sample_leaf={min_sample_leaf[3]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[0]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[1]},min_split={min_split[1]},min_sample_leaf={min_sample_leaf[1]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[2]},min_split={min_split[2]},min_sample_leaf={min_sample_leaf[2]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[3]},min_split={min_split[3]},min_sample_leaf={min_sample_leaf[3]})",
    ]
    counter = 0

    c = [1, 2, 3, 4, 5]
    for k in range(len(criterion)):
        for i in range(len(max_depth)):
            DecisionTree_metrics(
                weightedInput=weighted_input,
                label=label,
                max_depth=max_depth[i],
                sample_leaf=min_sample_leaf[i],
                sample_split=min_split[i],
                fileName=filenames[counter],
            )
            counter += 1


def Best_Configuration_Comparison(weighted_input, label):
    (
        _,
        DT_test_acc,
        _,
        DT_test_recall,
        _,
        DT_test_precision,
        _,
        _,
    ) = DecisionTree_metrics(
        weightedInput=weighted_input,
        label=label,
        max_depth=4,
        sample_leaf=10,
        sample_split=4,
        fileName="Best Decision Tree(depth=4, min_sample_leaf=10, split=4)",
    )

    (
        _,
        LDA_test_acc,
        _,
        LDA_test_recall,
        _,
        LDA_test_precision,
        _,
        _,
    ) = LDA_metrics(
            weightedInput=weighted_input,
            label=label,
            solver='svd',
            fileName="Best LDA(svd, prior=[0.36,0.64])",
        )
       (
        _,
        SVM_test_acc,
        _,
        SVM_test_recall,
        _,
        SVM_test_precision,
        _,
        _,
    ) = SVM_metrics(
                weightedInput=weighted_input,
                label=label,
                kernel='rbf',
                C=1,
                fileName="Best SVM(rbf, c=1)",
            )
    comparison(
        [
        DT_test_acc,
        LDA_test_acc,
        SVM_test_acc,
        DT_test_precision,
        LDA_test_precision,
        SVM_test_precision,
        DT_test_recall,
        LDA_test_recall
        SVM_test_recall
        ],
        [
        "DT_test_acc",
        "LDA_test_acc",
        "SVM_test_acc",
        "DT_test_precision",
        "LDA_test_precision",
        "SVM_test_precision",
        "DT_test_recall",
        "LDA_test_recall"
        "SVM_test_recall"
        ],
        "Comparison")
