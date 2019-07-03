import argparse
import random
import pandas as pd
import numpy as np
import tqdm

from collections import defaultdict, Counter
from .configs import dataset_map


def make_folds(n_folds) -> pd.DataFrame:
    df = pd.read_csv(dataset_map['train'], engine='python')

    cls_counts = Counter([classes for classes in df['diagnosis']])
    fold_cls_counts = defaultdict()
    for class_index in cls_counts.keys():
        fold_cls_counts[class_index] = np.zeros(4, dtype=np.int)

    df['fold'] = -1
    pbar = tqdm.tqdm(total=len(df))

    def get_fold(row):
        class_index = row['diagnosis']
        counts = fold_cls_counts[class_index]
        fold = np.argmin(counts)
        counts[fold] += 1
        fold_cls_counts[class_index] = counts
        row['fold']=fold
        pbar.update()
        return row
    
    df = df.apply(get_fold, axis=1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_csv('folds.csv', index=None)


if __name__ == '__main__':
    main()
