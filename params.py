import argparse

import yaml

def get_default_args():
    parser = get_parser()
    p = parser.parse_args([])

    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    return parser.parse_args([])

def get_parser():
    parser = argparse.ArgumentParser(description='KMeans Image Classification')
    parser.add_argument('--config', default=None)
    parser.add_argument('--metric', type=str, choices=['cosine', 'euclidean', 'manhattan', 'gaussian'], default='cosine')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--init_method', type=str, choices=['random', 'kmeans++'], default='random')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--visualization', type=bool, default=False)

    parser.add_argument('--save', type=str, default=None)

    return parser


def get_args():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    return parser.parse_args()

if __name__ == "__main__":
    get_parser().print_help()