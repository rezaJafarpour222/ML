from runner import DecisionTree_metrics, LDA_metrics, SVM_metrics


def SVM_Scenario(weighted_input, label):
    kernels = ["linear", "rbf"]
    c = [1, 2, 3, 4, 5]
    filenames = [
        f"SVM({kernels[0]}, c={c[0]})",
        f"SVM({kernels[0]}, c={c[1]})",
        f"SVM({kernels[0]}, c={c[2]})",
        f"SVM({kernels[0]}, c={c[3]})",
        f"SVM({kernels[0]}, c={c[4]})",
        f"SVM({kernels[1]}, c={c[0]})",
        f"SVM({kernels[1]}, c={c[1]})",
        f"SVM({kernels[1]}, c={c[2]})",
        f"SVM({kernels[1]}, c={c[3]})",
        f"SVM({kernels[1]}, c={c[4]})",
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
                title=filenames[counter],
            )
            counter += 1


def LDA_Scenario(weighted_input, label):
    solvers = ["svd", "lsqr", "eigen"]
    filenames = [
        f"LDA({solvers[0]})",
        f"LDA({solvers[1]})",
        f"LDA({solvers[2]})",
    ]
    counter = 0

    for k in range(len(solvers)):
        LDA_metrics(
            weightedInput=weighted_input,
            label=label,
            solver=solvers[k],
            fileName=filenames[counter],
            title=filenames[counter],
        )
        counter += 1


def DecisionTree_Scenario(weighted_input, label):
    criterion = ["entropy", "gini", "log_loss"]
    max_depth = [4, 5, 6, 10]
    min_split = [5, 6, 7, 8]
    min_sample_leaf = [5, 8, 2, 5]
    filenames = [
        f"Decision Tree({criterion[0]} max_depth={max_depth[0]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[0]} max_depth={max_depth[1]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[0]} max_depth={max_depth[2]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[0]} max_depth={max_depth[3]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[0]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[1]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[2]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[1]} max_depth={max_depth[3]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[0]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[1]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[2]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
        f"Decision Tree({criterion[2]} max_depth={max_depth[3]},min_split={min_split[0]},min_sample_leaf={min_sample_leaf[0]})",
    ]
    counter = 0

    c = [1, 2, 3, 4, 5]
    for k in range(len(criterion)):
        for i in range(len(max_depth)):
            DecisionTree_metrics(
                weightedInput=weighted_input,
                label=label,
                max_depth=max_depth[i],
                sample_leaf=min_sample_leaf[0],
                sample_split=min_split[0],
                fileName=filenames[counter],
                title=filenames[counter],
            )
            counter += 1
