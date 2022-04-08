# KMeans

计95 刘玉河 2019011560 liuyuhe19@mails.tsinghua.edu.cn

## File Structure

```text
kmeans
├── data
│   └── MNIST
├── model.py
├── params.py
├── README.md
├── requirements.txt
├── similarity.py
├── test.py
└── visualization.py
```

## How to run the code
1. Install packages in `requirements.txt`
2. run `python3 model.py [options]` to train a specified model, details and examples below.
3. run `python3 test.py` to run scripted multiple specifications.

```text
usage: model.py [-h] [--config CONFIG] [--metric {cosine,euclidean,manhattan,gaussian}] [--K K] [--init_method {random,kmeans++}] [--seed SEED] [--visualization VISUALIZATION] [--save SAVE]

KMeans Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --config            specify config file, default to None
  --metric            {cosine,euclidean,manhattan,gaussian}
  --K                 K
  --init_method       {random,kmeans++}
  --seed              random seed
  --visualization     whether to visually show the cluster results.
  --save              filename to save result
  --save_fig          filename to save visualization result, None to display directly
```

Example

Run cosine_kmeans++ with seed 0, visualize the result.
```bash
python3 model.py --metric cosine --init_method kmeans++ --seed 0 --visualization True
```

Run eight specifications (Do this to replay the results in the report)
```bash
python3 test.py
```