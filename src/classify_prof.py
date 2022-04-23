


import debias
import numpy as np
import random
from sklearn.svm import LinearSVC, SVC
import argparse
import matplotlib
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression

matplotlib.rcParams['agg.path.chunksize'] = 10000

import warnings
warnings.filterwarnings("ignore")

import pickle
from collections import defaultdict, Counter
from typing import List, Dict


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


def load(lang, p2i, repr_name, align):

    train = load_dataset("../data/biasbios/{}/train.pickle".format(lang))
    dev = load_dataset("../data/biasbios/{}/dev.pickle".format(lang))
    test = load_dataset("../data/biasbios/{}/test.pickle".format(lang))
    counter = count_profs_and_gender(train + dev + test)

    path = "../data/bert_encode_biasbios/{}/".format(lang)
    if align:
        x_train = np.load(path + "x_train_align.npy")
        x_dev = np.load(path + "x_dev_align.npy")
        x_test = np.load(path + "x_test_align.npy")
    elif not repr_name:
        x_train = np.load(path + "train_avg_mbert.npy")
        x_dev = np.load(path + "dev_avg_mbert.npy")
        x_test = np.load(path + "test_avg_mbert.npy")
    else:
        x_train = np.load(path + "train_{}.npy".format(repr_name))
        x_dev = np.load(path + "dev_{}.npy".format(repr_name))
        x_test = np.load(path + "test_{}.npy".format(repr_name))

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


def train_clf(x_train, y_train, x_test, y_test):
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression()

    #warm_start = True, penalty = 'l2',
    #                         solver = "saga", multi_class = 'multinomial', fit_intercept = False,
    #                         verbose = 5, n_jobs = 90, random_state = 1, max_iter = 7)

    clf.fit(x_train, y_train)
    score_test = clf.score(x_test, y_test)
    score_train = clf.score(x_train, y_train)
    print(clf.score(x_test, y_test))
    print(clf.score(x_train, y_train))

    return clf, score_test, score_train


def main():

    random.seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="EN", help="language")
    parser.add_argument("--inlp_mat", help="inlp matrix to load")
    parser.add_argument("--repr_name", help="name of repr to load")
    parser.add_argument("--align", action="store_true", help="use aligned repr")

    args = parser.parse_args()



    lang = args.lang
    p2i, i2p = load_dictionary("../data/biasbios/{}/profession2index.txt".format(lang))
    #g2i, i2g = load_dictionary("../data/biasbios/{}/gender2index.txt".format(lang))

    dev, train, test, x_train, x_dev, x_test, y_train, y_dev, y_test = load(lang, p2i, args.repr_name, args.align)

    print("before")
    clf_before, score_test_before, score_train_before = train_clf(x_train, y_train, x_test, y_test)
    if not args.inlp_mat:
        mat_used = "nomat"
        pickle.dump(clf_before, open("../data/classifiers/clf_prof_{}_before_{}".format(args.lang, mat_used), "wb"))
        with open("../data/classifiers/results.txt", "a") as f:
            f.write("classify prof, params:" + str(args) + "\n")
            f.write("score_test_before:" + str(score_test_before)+ "\n")
            f.write("score_train_before:" + str(score_train_before)+ "\n")
        return


    P = np.load(args.inlp_mat)

    x_train_inlp = x_train.dot(P)
    x_test_inlp = x_test.dot(P)

    print("after")
    clf_after, score_test_after, score_train_after = train_clf(x_train_inlp, y_train, x_test_inlp, y_test)

    mat_used = args.inlp_mat.rsplit("/",1)[1].split(".")[0]

    pickle.dump(clf_before, open("../data/classifiers/clf_prof_{}_before_{}".format(args.lang, mat_used), "wb"))
    pickle.dump(clf_after, open("../data/classifiers/clf_prof_{}_after_{}".format(args.lang, mat_used), "wb"))

    with open("../data/classifiers/results.txt", "a") as f:
        f.write("classify prof, params:" + str(args) + "\n")
        f.write("score_test_before:" + str(score_test_before)+ "\n")
        f.write("score_test_after:" + str(score_test_after)+ "\n")
        f.write("score_train_before:" + str(score_train_before)+ "\n")
        f.write("score_train_after:" + str(score_train_after)+ "\n")

    with open("../data/classifiers/results_prof.txt", "a") as f:
        f.write(args.inlp_mat + "\n")
        f.write(args.lang + "\n")
        f.write(args.repr_name + "\n")
        f.write(str(score_test_before)+ "\n")
        f.write(str(score_test_after)+ "\n")
        f.write("\n")

if __name__ == '__main__':
    main()