import os
import time

import numpy as np
import torch
from sklearn import metrics
from torchvision import datasets

import params
from visualization import visualization, kmeans_convergence, draw_number


class KMeans:
    def __init__(self, args):
        self.cluster_center = None
        self.cluster_label = None
        self.metric = args.metric
        self.init_method = args.init_method

        self.K = args.K

        if args.seed is not None:
            self.seed = args.seed
            torch.manual_seed(args.seed)
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(0,1000)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        self.delta_sequence = []

    def __str__(self):
        return f"{self.metric}_{self.init_method}_{self.K}_{self.seed}"

    def similarity(self, x, y):
        """
        :param x: nd batches of [756]
        :param y: nc batches of [756]
        :return: similarity between x and y
        """
        if self.metric == 'gaussian':
            return metrics.pairwise.rbf_kernel(x,y,1.0/784/256) #metrics.pairwise_distances(x,y, self.metric)
        elif self.metric == 'cosine':
            return metrics.pairwise.cosine_similarity(x,y)
        elif self.metric == 'euclidean':
            a = metrics.pairwise.pairwise_distances(x,y,'euclidean')
            return 1 / (a / np.max(a) + 0.0001)
        elif self.metric == 'manhattan':
            a = metrics.pairwise.pairwise_distances(x, y, 'manhattan')
            return 1 / (a / np.max(a) + 0.0001)
        # return cosine_similarity(x, y)

    def mean(self, cluster):
        return torch.mean(cluster.float(),dim=0)

    def initial_center(self, data):
        if self.init_method == 'random':    # random select k points from given n data points
            return data[torch.randint(len(data), (self.K,))]
        elif self.init_method == 'kmeans++':
            remain = set(range(len(data)))
            first = np.random.randint(0, len(data))
            remain.remove(first)
            chosen = [first]
            while len(chosen) < self.K:
                remain_list = list(remain)
                if self.metric == 'manhattan' or self.metric == 'euclidean':
                    P = np.square(np.max(metrics.pairwise_distances(data[remain_list],data[chosen], self.metric), axis=1))
                    P = P / sum(P)
                else:
                    sim = self.similarity(data[remain_list], data[chosen])
                    distance = np.max(sim) - sim    # [0, min+max]
                    P = np.square(np.max(distance, axis=1))
                    P = P/sum(P)
                next_point = np.random.choice(remain_list,p=P)
                remain.remove(next_point)
                chosen.append(next_point)
            print("Kmeans chose: ", chosen)
            return data[chosen]


    def train(self, data, label):
        data = self.preprocess(data)        # [60000, 784]

        # Randomly choose K center points
        center = self.initial_center(data).float()   #[10, 784]
        clusters = [[] for i in range(self.K)]
        self.delta_sequence = []

        iter_cnt = 0
        for t in range(0, 200):
            for i in range(self.K):
                clusters[i].clear()

            nearest = np.argmax(self.similarity(data, center), axis=1)  # find argmax w.r.t. each data
            for idx, _ in enumerate(data):
                clusters[nearest[idx]].append(idx)

            # compute new mean points for the clusters
            new_center = torch.stack([self.mean(data[cluster]) for cluster in clusters])

            if torch.equal(new_center, center):
                break                   # if center is fixed then break
            else:
                delta = torch.norm(new_center - center)
                self.delta_sequence.append(delta)
                print(f"\riteration {t} done, with center delta: {delta:.4f}", end='')
                iter_cnt += 1
                center = new_center     # else update the center
        print(f"\nStop at iter {iter_cnt}")

        # infer label for each cluster
        self.cluster_center = center
        cluster_labels = [label[cluster] if cluster else torch.tensor(-1) for cluster in clusters]
        self.cluster_label = torch.tensor([torch.mode(cluster_label).values for cluster_label in cluster_labels])


    def eval(self, test):
        assert self.cluster_center is not None, "This model hasn't been trained yet!"
        return self.cluster_label[
            np.argmax(self.similarity(self.preprocess(test), self.cluster_center), axis=1)]

    def preprocess(self, data):
        return torch.flatten(data, start_dim=1)


def main(args):
    model = KMeans(args)
    train_data = datasets.MNIST(root="./data/", train=True, download=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True)
    start_time = time.time()
    model.train(train_data.data, train_data.targets)
    end_time = time.time()
    print(f"Train time: {end_time - start_time:.2f} s")

    print(f"cluster labels: {model.cluster_label}")

    target = test_data.targets
    predicted = model.eval(test_data.data)
    acc = metrics.accuracy_score(target, predicted)
    print(f"Accuracy: {acc:.4f}")

    # Visualization
    cluster_fig = os.path.join(args.save_fig, str(model) + "_cluster.png") if args.save_fig else args.save_fig
    iter_fig = os.path.join(args.save_fig, str(model) + "_iter.png") if args.save_fig else args.save_fig
    if args.visualization:
        vis_target = train_data.targets[:5000]
        vis_pred = model.eval(train_data.data[:5000])
        visualization(train_data.data[:5000], vis_target, vis_pred, cluster_fig)
        kmeans_convergence(model.delta_sequence, iter_fig)
        for center in model.cluster_center:
            draw_number(center)

    return acc, len(model.delta_sequence)


if __name__ == "__main__":
    args = params.get_args()
    args.metric = 'euclidean'
    main(args)
