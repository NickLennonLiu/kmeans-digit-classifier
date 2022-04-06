import model
import params

def main(args):
    init_methods = ['random', 'kmeans++']
    metrics = ['cosine', 'euclidean', 'manhattan', 'gaussian']  # '

    for metric in metrics:
        for init_method in init_methods:
            args.metric = metric
            args.init_method = init_method

            acc_list = []
            k_list = []
            for seed in [0, 10, 20, 30, 40]:
                args.seed = seed
                acc, k = model.main(args)
                acc_list.append(acc)
                k_list.append(k)

            acc = sum(acc_list) / len(acc_list)
            k = sum(k_list) / len(k_list)
            with open('result.txt', 'a+') as f:
                print(f"{metric} {init_method}: {acc}, {k}", file=f)

def compare_random_with_kmeanspp(args):
    args.metric = 'cosine'
    args.save_fig = './workdir'
    args.seed = 40
    args.visualization = True

    args.init_method = 'random'
    model.main(args)

    args.init_method = 'kmeans++'
    model.main(args)


if __name__ == '__main__':
    args = params.get_args()
    # main(args)
    compare_random_with_kmeanspp(args)