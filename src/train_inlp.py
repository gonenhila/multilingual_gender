

import debias
import numpy as np
import random
from sklearn.svm import LinearSVC, SVC
import argparse
import matplotlib
import json
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression

matplotlib.rcParams['agg.path.chunksize'] = 10000

import warnings
warnings.filterwarnings("ignore")

import pickle
from collections import defaultdict, Counter
from typing import List, Dict

import time


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_dictionary(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    k2v, v2k = {}, {}
    for line in lines:
        k, v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k

    return k2v, v2k


def count_profs_and_gender(data: List[dict]):
    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1

    return counter


def load(lang, p2i, layer, embed_type, no_gender):

    train = load_dataset("../data/biasbios/{}/train.pickle".format(lang))
    dev = load_dataset("../data/biasbios/{}/dev.pickle".format(lang))
    test = load_dataset("../data/biasbios/{}/test.pickle".format(lang))
    counter = count_profs_and_gender(train + dev + test)

    path = "../data/bert_encode_biasbios/{}/".format(lang)
    if no_gender:
        path = "../data/bert_encode_biasbios/{}/wo_gender/".format(lang)

    if not layer:
        if not embed_type or embed_type=="cls":
            x_train = np.load(path + "train_cls_mbert.npy")
            x_dev = np.load(path + "dev_cls_mbert.npy")
            x_test = np.load(path + "test_cls_mbert.npy")
        else:
            assert(embed_type=="avg")
            x_train = np.load(path + "train_avg_mbert.npy")
            x_dev = np.load(path + "dev_avg_mbert.npy")
            x_test = np.load(path + "test_avg_mbert.npy")

    else:
        if embed_type:
            x_train = np.load(path + "train_{}_layer{}_mbert.npy".format(embed_type, layer))
            x_dev = np.load(path + "dev_{}_layer{}_mbert.npy".format(embed_type, layer))
            x_test = np.load(path + "test_{}_layer{}_mbert.npy".format(embed_type, layer))
        else:
            raise ValueError("please indicate type")

    assert len(train) == len(x_train)
    assert len(dev) == len(x_dev)
    assert len(test) == len(x_test)

    f, m = 0., 0.
    prof2fem = dict()

    for k, values in counter.items():
        f += values['f']
        m += values['m']
        prof2fem[k] = values['f'] / (values['f'] + values['m'])

    print(f / (f + m))
    print(prof2fem)

    y_train = np.array([p2i[entry["p"]] for entry in train])
    y_dev = np.array([p2i[entry["p"]] for entry in dev])
    y_test = np.array([p2i[entry["p"]] for entry in test])

    return dev, train, test, x_train, x_dev, x_test, y_train, y_dev, y_test



def get_projection_matrix(num_clfs, X_train, Y_train_gender, X_dev, Y_dev_gender, Y_train_task, Y_dev_task):

    is_autoregressive = True
    min_acc = 0.
    dim = 768
    n = num_clfs
    start = time.time()

    gender_clf = SGDClassifier
    params = {'loss':'hinge', 'fit_intercept':True, 'max_iter':3000000, 'tol':1e-4, 'n_iter_no_change':600, 'n_jobs':16}

    P, rowspace_projections, Ws, bs, iters = debias.get_debiasing_projection_iters(gender_clf, params, n, dim, is_autoregressive,
                                                                  min_acc,
                                                                  X_train, Y_train_gender, X_dev, Y_dev_gender,
                                                                  Y_train_main=Y_train_task, Y_dev_main=Y_dev_task,
                                                                  by_class=False)

    print("time: {}".format(time.time() - start))
    return P, rowspace_projections, Ws, bs, iters




def main():

    random.seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="EN", help="language")
    parser.add_argument("--layer", help="layer to use")
    parser.add_argument("--type", help="cls or avg")
    parser.add_argument("--iters", type=int, default=300, help="num of iterations for INLP")
    parser.add_argument("--output_path", help="where to save P")
    parser.add_argument("--no_gender", action="store_true")

    args = parser.parse_args()

    lang = args.lang.upper()
    p2i, i2p = load_dictionary("../data/biasbios/{}/profession2index.txt".format(lang))
    g2i, i2g = load_dictionary("../data/biasbios/{}/gender2index.txt".format(lang))

    if args.no_gender:
        args.output_path += "wo_gender/"

    dev, train, test, x_train, x_dev, x_test, y_train, y_dev, y_test = load(lang, p2i, args.layer, args.type, args.no_gender)
    num_clfs = args.iters
    y_dev_gender = np.array([g2i[d["g"]] for d in dev])
    y_train_gender = np.array([g2i[d["g"]] for d in train])

    P, rowspace_projections, Ws, bs, iters = get_projection_matrix(num_clfs, x_train, y_train_gender, x_dev, y_dev_gender, y_train,
                                                        y_dev)

    # Save to file

    if not args.layer:
        if not args.type or args.type == "cls":
            # normal - last cls, no indication in the names
            np.save(args.output_path + "P_mbert_{}_{}.npy".format(args.iters, lang), P)
            np.save(args.output_path + "Ws_mbert_{}_{}.npy".format(args.iters, lang), Ws)
            np.save(args.output_path + "bs_mbert_{}_{}.npy".format(args.iters, lang), bs)
            with open(args.output_path + "iters_mbert_{}_{}.json".format(args.iters, lang), "w") as f:
                json.dump(iters, f)

        else:
            assert(args.type == "avg")
            # last avg, no indication of layer in the names
            np.save(args.output_path + "P_mbert_{}_{}_avg.npy".format(args.iters, lang), P)
            np.save(args.output_path + "Ws_mbert_{}_{}_avg.npy".format(args.iters, lang), Ws)
            np.save(args.output_path + "bs_mbert_{}_{}_avg.npy".format(args.iters, lang), bs)
            with open(args.output_path + "iters_mbert_{}_{}_avg.json".format(args.iters, lang), "w") as f:
                json.dump(iters, f)
    else:
        if args.type:
            np.save(args.output_path + "P_mbert_{}_{}_{}_layer{}.npy".format(args.iters, lang, args.type, args.layer), P)
            np.save(args.output_path + "Ws_mbert_{}_{}_{}_layer{}.npy".format(args.iters, lang, args.type, args.layer), Ws)
            np.save(args.output_path + "bs_mbert_{}_{}_{}_layer{}.npy".format(args.iters, lang, args.type, args.layer), bs)
            with open(args.output_path + "iters_mbert_{}_{}_{}_layer{}.json".format(args.iters, lang, args.type, args.layer), "w") as f:
                json.dump(iters, f)
        else:
            raise ValueError("please indicate type")


if __name__ == '__main__':
    main()