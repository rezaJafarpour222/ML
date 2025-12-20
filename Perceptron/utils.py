import numpy as np
import matplotlib

matplotlib.use("GTK3Agg")  # this line is for linux comment it for windows

import matplotlib.pyplot as plt


def z_score_scaler(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_Scaled = (X - X_mean) / X_std
    return X_Scaled


def splitter(splitPrecent, data, label):
    np.random.seed(1000)
    data_indices = np.arange(data.shape[0])
    np.random.shuffle(data_indices)

    data_splitter_idx = int(splitPrecent * data.shape[0])
    splitted_data = data[data_indices[:data_splitter_idx]]
    splitted_label = label[data_indices[:data_splitter_idx]]

    splitted_indices = np.arange(splitted_data.shape[0])

    split_idx = int(0.8 * splitted_data.shape[0])

    X_train = splitted_data[splitted_indices[:split_idx]]
    Y_train = splitted_label[splitted_indices[:split_idx]]
    X_test = splitted_data[splitted_indices[split_idx:]]
    Y_test = splitted_label[splitted_indices[split_idx:]]

    return (X_train, Y_train, X_test, Y_test)


def plotter(
    x_values,
    values_arr,
    line_label_arr,
    file_name,
    x_label,
    y_label,
    title="default",
    show_percentage_for_x=False,
    show_values_for_each=False,
):
    plt.figure(figsize=(16, 9), dpi=100)
    line_List = []
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["o", "*", "x", "D", "v", "s", "^", "+", "p", "h"]

    for i in range(len(values_arr)):
        (line,) = plt.plot(
            x_values,
            values_arr[i],
            label=line_label_arr[i],
            marker=markers[i],
            color=colors[i],
        )
        line_List.append(line)
        if show_values_for_each:
            for x, y in zip(x_values, values_arr[i]):
                plt.text(
                    x,
                    y,
                    f"{y:.2f}",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    color=colors[i],
                    alpha=0.7,
                )

    if show_percentage_for_x:
        plt.xticks(x_values, [f"{int(x*100)}%" for x in x_values])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(handles=line_List, loc="best")
    plt.tight_layout()
    plt.savefig(f"Perceptron/plots/{file_name}", dpi=500)
    plt.close()
    # plt.show()
