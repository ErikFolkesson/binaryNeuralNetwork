import matplotlib.pyplot as plt
import torch


def plot_mnist_data(data: torch.utils.data.Dataset, n_rows=4, n_cols=4):
    # Plot more images
    torch.manual_seed(99)
    fig = plt.figure(figsize=(9, 9))
    rows, cols = n_rows, n_cols
    class_names = data.classes
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(data), size=[1]).item()
        img, label = data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)
