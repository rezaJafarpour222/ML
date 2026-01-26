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
    np.random.seed(6)
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


def plotter(values_arr, label_arr, file_name, y_label, title="default", width=0.2):
    plt.figure(figsize=(16, 9), dpi=100)
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
    bars = plt.bar(label_arr, values_arr, color=colors, width=width)
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{y * 100:.1f}%",  # ← just show the value
            ha="center",
            va="bottom",
        )
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    plt.ylim(0, 1.05)
    plt.title(title)
    # plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(False, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"3rdAssignment/plots/{file_name}", dpi=500)
    plt.close()
    # plt.show()
