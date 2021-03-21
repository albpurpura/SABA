import os

import matplotlib.pyplot as plt
import numpy as np


def load_data(fpath=''):
    if len(fpath) == 0:
        fpaths = ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']
    else:
        fpaths = fpath

    honest_data = []
    dishonest_data = []
    for fpath in fpaths:
        header = True
        for line in open(fpath):
            data = line.strip().split(',')
            if header:
                header = False
                continue
            is_honest = data[-1] == 'H'
            answers = np.array(data[:10])
            if is_honest:
                honest_data.append(answers)
            else:
                dishonest_data.append(answers)
    return np.array(honest_data), np.array(dishonest_data)


def evaluate_pair(true_lies_mask, detected_lies_mask):
    if np.sum(detected_lies_mask) > 0:
        prec = np.sum(
            np.where((detected_lies_mask == true_lies_mask) & (true_lies_mask == np.ones_like(true_lies_mask)),
                     np.ones_like(detected_lies_mask), 0)) / np.sum(detected_lies_mask)
    else:
        prec = 0.
    if np.sum(true_lies_mask) > 0:
        rec = np.sum(np.where((detected_lies_mask == true_lies_mask) & (true_lies_mask == np.ones_like(true_lies_mask)),
                              np.ones_like(detected_lies_mask), 0)) / np.sum(true_lies_mask)
    else:
        rec = 0.

    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    return prec, rec, f1


def compute_precs_recs_f1s(true_lies_mask, detected_lies_mask):
    curr_precs = []
    curr_f1s = []
    curr_recs = []
    for i in range(detected_lies_mask.shape[0]):
        prec, rec, f1 = evaluate_pair(true_lies_mask[i], detected_lies_mask[i])
        curr_f1s.append(f1)
        curr_precs.append(prec)
        curr_recs.append(rec)
    # return np.mean(curr_precs), np.mean(curr_recs), np.mean(curr_f1s)
    return curr_precs, curr_recs, curr_f1s


def estimate_per_question_thresholds(tfidf_scores, thr=80):
    return [np.percentile(tfidf_scores[:, j], thr) for j in range(tfidf_scores.shape[1])]


def optimize_thr_dist(honest_data, faked_data, true_lies_mask):
    perfs = []
    thrs = []
    for percentile in np.arange(50, 100, step=1):
        thresholds = estimate_per_question_thresholds(honest_data, percentile)
        detected_lies_mask = np.array(
            [np.where(faked_data[:, j] >= thresholds[j], np.ones_like(faked_data[:, j]), 0) for j in
             range(faked_data.shape[1])]).transpose()
        # p, r, f1 = compute_precs_recs_f1s(true_lies_mask, detected_lies_mask)
        acc = np.mean(np.where(true_lies_mask == detected_lies_mask, np.ones_like(detected_lies_mask), 0))
        perfs.append(acc)
        thrs.append(percentile)
    best_thr = thrs[np.argmax(perfs)]
    print('best thr={}'.format(best_thr))
    return best_thr


def get_train_test_data(test_set):
    all_fpaths = ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']
    # fnames_mapper = {'data/BF_CTU.csv': 'C', 'data/BF_V.csv': 'S', 'data/BF_OU.csv': 'H'}
    all_fpaths.remove(test_set)
    assert len(all_fpaths) == 2
    hdata_train, ldata_train = load_data(all_fpaths)
    hdata_train = np.array(hdata_train, dtype=np.float)
    ldata_train = np.array(ldata_train, dtype=np.float)

    hdata_test, ldata_test = load_data([test_set])
    hdata_test = np.array(hdata_test, dtype=np.float)
    ldata_test = np.array(ldata_test, dtype=np.float)
    return hdata_train, hdata_test, ldata_train, ldata_test


def compute_perf_distr_model(test_set):
    print('Distribution Model')
    hdata_train, hdata_test, ldata_train, ldata_test = get_train_test_data(test_set)
    true_lies_mask_train = np.where(hdata_train != ldata_train, np.ones_like(hdata_train), 0)
    true_lies_mask_test = np.where(hdata_test != ldata_test, np.ones_like(ldata_test), 0)
    percentile = optimize_thr_dist(hdata_train, ldata_train, true_lies_mask_train)
    print('percentile: {}'.format(percentile))
    thresholds = estimate_per_question_thresholds(hdata_train, percentile)
    detected_lies_mask_test = np.array(
        [np.where(hdata_test[:, j] >= thresholds[j], np.ones_like(hdata_test[:, j]), 0) for j in
         range(hdata_test.shape[1])]).transpose()
    p, r, f1 = compute_precs_recs_f1s(true_lies_mask_test, detected_lies_mask_test)
    p = np.mean(p)
    r = np.mean(r)
    f1 = np.mean(f1)
    print('Prec: {}, Recall: {}, F1Score: {}'.format(p, r, f1))
    print('{:.4f} & {:.4f} & {:.4f}'.format(p, r, f1))


def get_k_closest_vecs(v, pool, k):
    dists = [np.sum(np.dot(v, p) / (np.linalg.norm(v, ord=2) * np.linalg.norm(p, ord=2))) for p in pool]
    return np.array(pool)[np.argsort(dists)[: k]]


def compute_detected_lies_mask(v, closest_vecs, thr=1.0):
    avg_neighbor = np.mean(closest_vecs, axis=0)
    assert avg_neighbor.shape == closest_vecs[0].shape
    return np.where(np.abs(avg_neighbor - v) > thr, np.ones_like(v), 0)


def compute_perf_knn_model(test_set):
    print('k-NN Model')
    fnames_mapper = {'data/BF_CTU.csv': 'C', 'data/BF_V.csv': 'S', 'data/BF_OU.csv': 'H'}
    hdata_train, hdata_test, ldata_train, ldata_test = get_train_test_data(test_set)

    true_lies_mask_train = np.where(hdata_train != ldata_train, np.ones_like(hdata_train), 0)
    true_lies_mask_test = np.where(hdata_test != ldata_test, np.ones_like(ldata_test), 0)

    perfs = []
    thrs = []
    ks = []
    for thr in np.arange(1, 5, 1):
        for k in np.arange(1, 50, 5):
            detected_lies_masks_train = []
            for v in ldata_train:
                closest_vecs = get_k_closest_vecs(v, hdata_train, k)
                curr_detected_lies = compute_detected_lies_mask(v, closest_vecs, thr=thr)
                detected_lies_masks_train.append(curr_detected_lies)
            detected_lies_masks_train = np.array(detected_lies_masks_train)
            p, r, f1 = compute_precs_recs_f1s(true_lies_mask_train, detected_lies_masks_train)
            p = np.mean(p)
            r = np.mean(r)
            f1 = np.mean(f1)
            perfs.append(p)
            ks.append(k)
            thrs.append(thr)

    best_k = ks[np.argmax(perfs)]
    best_thr = thrs[np.argmax(perfs)]
    print('best K: {}, best thr: {}'.format(best_k, best_thr))

    detected_lies_mask_test = []
    for v in ldata_test:
        closest_vecs = get_k_closest_vecs(v, hdata_test, best_k)
        curr_detected_lies = compute_detected_lies_mask(v, closest_vecs, thr=best_thr)
        detected_lies_mask_test.append(curr_detected_lies)
    detected_lies_mask_test = np.array(detected_lies_mask_test)
    p, r, f1 = compute_precs_recs_f1s(true_lies_mask_test, detected_lies_mask_test)
    print('Prec: {}, Recall: {}, F1Score: {}'.format(np.mean(p), np.mean(r), np.mean(f1)))
    print('{:.4f} & {:.4f} & {:.4f}'.format(np.mean(p), np.mean(r), np.mean(f1)))

    create_hist_by_n_lies(true_lies_mask_test, p, fname=fnames_mapper[test_set] + '_prec', measure_name='Precision',
                          opath='./output/figures/knn/')
    create_hist_by_n_lies(true_lies_mask_test, r, fname=fnames_mapper[test_set] + '_rec', measure_name='Recall',
                          opath='./output/figures/knn/')
    create_hist_by_n_lies(true_lies_mask_test, f1, fname=fnames_mapper[test_set] + '_f1', measure_name='F1 Score',
                          opath='./output/figures/knn/')


def compute_faked_answ_mask(pred_n_lies_per_test, tfidfs_faked_test, thresholds):
    new_lies_masks = []
    for i in range(tfidfs_faked_test.shape[0]):
        n_pred_lies = pred_n_lies_per_test[i]
        lies_indices = np.argsort(-np.where(tfidfs_faked_test[i] >= thresholds, tfidfs_faked_test[i], 0))[
                       :int(np.ceil(n_pred_lies))]
        curr_lies_mask = np.zeros_like(tfidfs_faked_test[i])
        for k in lies_indices:
            curr_lies_mask[k] = 1
        new_lies_masks.append(curr_lies_mask)
    return np.array(new_lies_masks)


def compute_per_question_accuracy(true_lies_mask, detected_lies_mask):
    per_question_accuracy = []
    for question_index in range(true_lies_mask.shape[1]):
        accuracy = np.mean(np.where(true_lies_mask[:, question_index] == detected_lies_mask[:, question_index],
                                    np.ones_like(detected_lies_mask[:, question_index]), 0))
        per_question_accuracy.append(accuracy)

    return np.array(per_question_accuracy)


def compute_n_lies_per_sample_dist(honest, dishonest):
    true_lies_mask = np.where(np.abs(honest - dishonest) > 0, np.ones_like(honest), 0)
    n_lies = np.sum(true_lies_mask, axis=-1)
    import collections
    counter = collections.Counter(n_lies)
    return counter


def create_hist_by_n_lies(true_lies_mask, per_test_perf, measure_name, fname, opath='./output/figures/tfidf/'):
    if not os.path.exists(opath):
        os.makedirs(opath)
    # n_lies_map = collections.Counter(np.sum(true_lies_mask, axis=-1))
    n_lies_map = {}
    for i in range(true_lies_mask.shape[0]):
        key = np.sum(true_lies_mask[i])
        if key not in n_lies_map.keys():
            n_lies_map[key] = [i]
        else:
            n_lies_map[key].append(i)

    per_test_perf = np.array(per_test_perf)
    y = []
    x = []
    gsize = []
    n_lies_all = sorted(list(n_lies_map.keys()))
    for n_lies in n_lies_all:
        if n_lies > 0:
            indices = n_lies_map[n_lies]
            group_acc = np.mean(per_test_perf[indices])
            x.append(n_lies)
            y.append(group_acc)
            gsize.append(len(indices) / len(true_lies_mask))

    plt.figure()
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
    ax = plt.gca()
    # plt.title('Faking detection {} by number of faked answers'.format(measure_name))
    plt.bar(x=x, height=y, label='{}'.format(measure_name))
    plt.xlabel('Number of faked answers per sample', fontsize=20)
    plt.ylabel('{}'.format(measure_name), fontsize=20)

    plt.plot(sorted(list(x)), gsize, color='r', label='Group sizes PDF')
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim(0, 1)
    plt.xlim(0.5, max(x) + 1)
    leg = ax.legend(prop={'size': 20})
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(opath + fname + '-{}-by-n-lies.png'.format(measure_name), bbox_inches='tight',
                pad_inches=0.01)


def create_perf_histogram(avgd_perfs_by_question, fname, measure_name, opath='./output/figures/tfidf/'):
    if not os.path.exists(opath):
        os.makedirs(opath)
    x = [i + 1 for i in range(len(avgd_perfs_by_question))]

    plt.figure()
    ax = plt.gca()
    plt.title('Faking detection {}'.format(measure_name))
    plt.bar(x=x, height=avgd_perfs_by_question, label=measure_name)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.savefig(opath + fname + '.png')


def run():
    for test_set in ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']:
        print(test_set)
        # compute_perf_distr_model(test_set)
        compute_perf_knn_model(test_set)


if __name__ == '__main__':
    run()
