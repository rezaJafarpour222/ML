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
    np.random.seed(42)
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
    first,
    x_label,
    second,
    y_label,
    file_name,
    title="default",
    first_line_title="first",
    second_line_title="second",
    show_percentage_for_x=False,
    show_values_for_each=False,
):
    plt.figure(figsize=(16, 9), dpi=500)

    (line1,) = plt.plot(
        x_values, first, label=first_line_title, marker="o", color="red"
    )
    (line2,) = plt.plot(
        x_values, second, label=second_line_title, marker="o", color="blue"
    )

    # Title + labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Format x-axis as percentages
    if show_percentage_for_x:
        plt.xticks(x_values, [f"{int(x*100)}%" for x in x_values])

    # Add text labels on the dots
    if show_values_for_each:
        for x, y in zip(x_values, first):
            plt.text(
                x,
                y,
                f"{y:.4f}",
                fontsize=9,
                ha="center",
                va="bottom",
                color="red",
                alpha=0.7,
            )
        for x, y in zip(x_values, second):
            plt.text(
                x,
                y,
                f"{y:.4f}",
                fontsize=9,
                ha="center",
                va="bottom",
                color="blue",
                alpha=0.7,
            )

    # Grid + legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(handles=[line1, line2], loc="best")
    plt.tight_layout()
    plt.savefig(f"LinearRegression/plots/{file_name}", dpi=500)
    plt.close()
