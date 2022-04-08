import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torchvision import datasets

def visualization(data, target, predicted, filename=None):
    """
    :param data:        data to cluster
    :param target:      true label
    :param predicted:   predicted label
    """
    data = torch.stack(list(map(torch.flatten, data)))   # preprocessing

    decomposer = TSNE(n_components=2)
    X_tsne = decomposer.fit_transform(data)

    target_label = [str(int(i.data)) for i in target]
    pred_label = [str(int(i.data)) for i in predicted]

    df_tsne = pd.DataFrame({'Dim1': X_tsne[:,0], 'Dim2': X_tsne[:,1], 'class':target_label})
    df_pred = pd.DataFrame({'Dim1': X_tsne[:, 0], 'Dim2': X_tsne[:, 1], 'class': pred_label})

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,8))
    sns.scatterplot(data=df_tsne, hue='class', hue_order=[str(i) for i in range(0,10)], x='Dim1', y='Dim2', ax=ax1)
    ax1.set_title("t-SNE")
    sns.scatterplot(data=df_pred, hue='class', hue_order=[str(i) for i in range(0,10)], x='Dim1', y='Dim2', ax=ax2)
    ax2.set_title("KMeans")

    if filename:
        fig.savefig(filename)
    else:
        fig.show()

def kmeans_convergence(sequence, filename=None):
    plt.figure(figsize=(8,4))
    plt.plot(np.log(sequence))
    plt.title("Delta of center (in log10)")
    plt.xlabel("Iteration")
    plt.ylabel("Delta{center}")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def draw_number(data):
    plt.imshow(data.resize(28,28), cmap="binary")

if __name__ == "__main__":
    test_data = datasets.MNIST(root="./data/", train=True, download=True)