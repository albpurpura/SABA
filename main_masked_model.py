import argparse
import fcntl
import logging
import os
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from correlation_analyses import questions_correlation_heatmap, create_latex_table_mean_std_by_answ, \
    approx_each_ans_with_closest_valid_value
from masked_model import AutoEncoder

tf.disable_v2_behavior()
flags = tf.app.flags
FLAGS = flags.FLAGS
# plot_all_figures = False
model_ckpt_path = './output/ckpts_' + str(uuid.uuid4())
exp_suff = 'masked-model'


# exp_suff = 'masked-model-no-sa'


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument("--log_file", type=str, default='./log_hparam_opt_masked_model_ptsd.txt')
    parser.add_argument("--log_perf", type=bool, default=False)
    parser.add_argument("--approximate_to_closest_valid_value", type=bool, default=False)
    parser.add_argument("--fnorm_strategy", type=str, default='MinMax')  # MinMax, None
    # parser.add_argument("--fpath", type=str, default='data/BF_CTU.csv')
    # parser.add_argument("--fpath", type=str, default='data/BF_V.csv')
    parser.add_argument("--fpath", type=str, default='data/BF_OU.csv')
    parser.add_argument("--n_epochs", type=int, default=1000)
    # params that could be tuned
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_masked_samples", type=int, default=5)
    parser.add_argument("--n_items_to_mask_on_inference", type=int, default=0)
    parser.add_argument("--hidd_size_enc", type=int, default=4)
    parser.add_argument("--femb_size", type=int, default=16)
    parser.add_argument("--hidd_size_sa", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lie_detection_eval_thr", type=float, default=0.2)


def set_best_hparams():
    if not FLAGS.log_perf:
        if FLAGS.fpath == 'data/BF_OU.csv':
            FLAGS.fnorm_strategy = 'MinMax'
            FLAGS.batch_size = 1
            FLAGS.n_masked_samples = 1
            FLAGS.n_items_to_mask_on_inference = 0
            FLAGS.hidd_size_enc = 2
            FLAGS.femb_size = 8
            FLAGS.hidd_size_sa = 4
            FLAGS.learning_rate = 0.001
            FLAGS.lie_detection_eval_thr = 0.2
        elif FLAGS.fpath == 'data/BF_V.csv':
            FLAGS.fnorm_strategy = 'MinMax'
            FLAGS.batch_size = 128
            FLAGS.n_masked_samples = 1
            FLAGS.n_items_to_mask_on_inference = 0
            FLAGS.hidd_size_enc = 2
            FLAGS.femb_size = 16
            FLAGS.hidd_size_sa = 8
            FLAGS.learning_rate = 0.001
            FLAGS.lie_detection_eval_thr = 0.2
        elif FLAGS.fpath == 'data/BF_CTU.csv':
            FLAGS.fnorm_strategy = 'MinMax'
            FLAGS.batch_size = 16
            FLAGS.n_masked_samples = 1
            FLAGS.n_items_to_mask_on_inference = 0
            FLAGS.hidd_size_enc = 2
            FLAGS.femb_size = 8
            FLAGS.hidd_size_sa = 4
            FLAGS.learning_rate = 0.01
            FLAGS.lie_detection_eval_thr = 0.2
        # FLAGS.hidd_size_enc = FLAGS.hidd_size_sa
        # FLAGS.hidd_size_sa = -1


def load_data(fpaths, n_questions=10):
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
            answers = np.array(data[:n_questions])
            if is_honest:
                honest_data.append(answers)
            else:
                dishonest_data.append(answers)
    return np.array(honest_data), np.array(dishonest_data)


def normalize_scores(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    else:
        return scaler.transform(data)


def generate_training_data(data, real_honest_seqs=None):
    data = np.array(data)
    masked_seqs = []
    true_seqs = []
    indices_masks = []
    special_mask_encoding = np.zeros(data.shape[-1] + 1)
    special_mask_encoding[-1] = 1.
    for i in range(len(data)):
        curr_im = np.zeros(len(data[i]))
        curr_masked_seq = np.zeros((data.shape[-1], data.shape[1] + 1))  # pseudo one-hot encoding
        for j in np.random.choice([idx for idx in range(len(data[i]))], FLAGS.n_masked_samples,
                                  replace=False):  # mask position j
            curr_im[j] = 1
            for k in range(curr_masked_seq.shape[0]):
                if j != k and curr_masked_seq[k, -1] == 0:
                    curr_masked_seq[k, k] = data[i, k]
                else:
                    curr_masked_seq[k] = special_mask_encoding

        masked_seqs.append(curr_masked_seq)
        indices_masks.append(curr_im)
        if real_honest_seqs is not None:
            true_seqs.append(real_honest_seqs[i])
        else:
            true_seqs.append(data[i])

    return np.array(masked_seqs), np.array(true_seqs), np.array(indices_masks)


def get_batches(data, batch_size=8):
    masked_seqs, true_seqs, indices_masks = generate_training_data(data)
    bms = []
    bts = []
    bim = []
    for i in range(len(masked_seqs)):
        bms.append(masked_seqs[i])
        bts.append(true_seqs[i])
        bim.append(indices_masks[i])

        if len(bms) == batch_size:
            yield bms, bts, bim
            bms = []
            bts = []
            bim = []

    if len(bms) > 0:
        yield bms, bts, bim


def test_model(sess, m, model_path, data, batch_size, seed):
    tf.set_random_seed(seed)
    m.saver.restore(sess, model_path)
    sess.graph.finalize()
    losses = []
    for bms, bts, bim in get_batches(data, batch_size=batch_size):
        loss = sess.run(m.loss, feed_dict={m.masked_data: bms,
                                           m.masked_indices_mask: bim,
                                           m.true_data: bts,
                                           m.training: False})
        losses.append(loss)
    return np.mean(losses)


def train_and_evaluate(sess, m, train_data, valid_data):
    max_patience = 20
    patience = max_patience

    best_val_loss = 1
    model_suffix = str(uuid.uuid4())
    if not os.path.exists(model_ckpt_path):
        os.makedirs(model_ckpt_path, exist_ok=True)
    elif FLAGS.log_perf:
        try:
            shutil.rmtree(model_ckpt_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    valid_losses = []
    model_paths = []

    early_stopping = False
    for epoch in range(FLAGS.n_epochs):
        if early_stopping:
            break
        for bms, bts, bim in get_batches(train_data, batch_size=FLAGS.batch_size):
            _, loss, step, rseq = sess.run([m.train_op, m.loss, m.global_step, m.reconstructed_seq],
                                           feed_dict={m.masked_data: bms,
                                                      m.masked_indices_mask: bim,
                                                      m.true_data: bts,
                                                      m.training: True})
            # rseq = rseq * bim
            if step % 10 == 0:
                step = sess.run(m.global_step)
                save_path = m.saver.save(sess, os.path.join(model_ckpt_path, 'ckpt_' + model_suffix), global_step=step)
                valid_loss = test_model(sess, m, save_path, valid_data, FLAGS.batch_size, FLAGS.seed)
                valid_losses.append(valid_loss)
                model_paths.append(save_path)
                if not FLAGS.log_perf:
                    print('epoch: {}, step: {}, loss: {}, validation loss: {}'.format(epoch, step, loss, valid_loss))
                if valid_loss < best_val_loss:
                    patience = max_patience
                    best_val_loss = valid_loss
                else:
                    patience -= 1
                    if patience <= 0:
                        early_stopping = True
                        break

    best_model_index = np.argmin(valid_losses)
    best_valid_loss = valid_losses[np.argmin(valid_losses)]
    return model_paths[best_model_index], best_valid_loss, sess, m


def lie_detection_eval(model_path, sess, m, test_h_data, test_l_data, thr):
    tf.set_random_seed(FLAGS.seed)
    m.saver.restore(sess, model_path)
    sess.graph.finalize()
    lie_detection_f1s = []
    precs = []
    recs = []
    detected_lies_masks_dishonest_honest_seqs = []
    preds_grouped_by_n_lies = {}
    for i in range(len(test_l_data)):
        curr_sample_n_lies = np.sum(
            np.where(np.abs(test_l_data[i] - test_h_data[i]) > 0, np.ones_like(test_l_data[i]), 0))
        if curr_sample_n_lies not in preds_grouped_by_n_lies.keys():
            preds_grouped_by_n_lies[curr_sample_n_lies] = {'prec': [], 'rec': [], 'f1': [], 'mae': []}

        sample_lie_detection_f1, prec, rec, detected_lies_mask, dishonest_sequence, resp_honest_seq = evaluate_pair_alt(
            test_l_data[i], test_h_data[i], sess, m, thr=thr)

        detected_lies_masks_dishonest_honest_seqs.append((detected_lies_mask, dishonest_sequence, resp_honest_seq))

        lie_detection_f1s.append(sample_lie_detection_f1)
        precs.append(prec)
        recs.append(rec)

        preds_grouped_by_n_lies[curr_sample_n_lies]['prec'].append(prec)
        preds_grouped_by_n_lies[curr_sample_n_lies]['rec'].append(rec)
        preds_grouped_by_n_lies[curr_sample_n_lies]['f1'].append(sample_lie_detection_f1)

    return np.mean(lie_detection_f1s), np.mean(precs), np.mean(recs), preds_grouped_by_n_lies, \
           detected_lies_masks_dishonest_honest_seqs


def evaluate_pair_alt(dishonest_sequence, resp_honest_seq, sess, m, thr, pred_n_lies_per_test=None):
    n_items_to_mask = FLAGS.n_items_to_mask_on_inference
    masked_seqs = []
    indices_masks = []
    special_mask_encoding = np.zeros(dishonest_sequence.shape[-1] + 1)
    special_mask_encoding[-1] = 1.
    n_samples = len(dishonest_sequence)
    for s in range(n_samples):
        curr_im = np.zeros(len(dishonest_sequence))
        curr_masked_seq = np.zeros(
            (dishonest_sequence.shape[-1], dishonest_sequence.shape[-1] + 1))  # pseudo one-hot encoding
        indices_to_mask = list(np.random.choice([i for i in range(len(dishonest_sequence))], n_items_to_mask))
        indices_to_mask.append(s)  # to guarantee to have a pred on every element of the seq
        indices_to_mask = set(indices_to_mask)
        for j in indices_to_mask:  # mask position j
            curr_im[j] = 1
            for k in range(curr_masked_seq.shape[0]):
                if j != k and curr_masked_seq[k, -1] == 0:
                    curr_masked_seq[k, k] = dishonest_sequence[k]
                else:
                    curr_masked_seq[k] = special_mask_encoding
        indices_masks.append(curr_im)
        masked_seqs.append(curr_masked_seq)
    predicted_masked_items_samples = [sess.run(m.reconstructed_seq, feed_dict={m.masked_data: masked_seqs,
                                                                               m.masked_indices_mask: indices_masks,
                                                                               m.training: False}) for _ in range(50)],
    predicted_masked_items_samples = np.array(predicted_masked_items_samples[0])
    confidence = np.std(predicted_masked_items_samples, axis=0)
    predicted_masked_items = np.mean(predicted_masked_items_samples, axis=0)
    reconstructed_sequence = np.sum(np.multiply(predicted_masked_items, indices_masks), axis=0) / np.sum(
        np.where(np.multiply(predicted_masked_items, indices_masks) > 0, np.ones_like(predicted_masked_items), 0),
        axis=0)
    confidence_by_item = np.sum(np.multiply(confidence, indices_masks), axis=0) / np.sum(
        np.where(np.multiply(confidence, indices_masks) > 0, np.ones_like(predicted_masked_items), 0),
        axis=0)
    if np.max(confidence_by_item) == 0:
        confidence_by_item = np.ones_like(confidence_by_item)
    normalized_confidence_by_item = softmax(1 / softmax(confidence_by_item, axis=-1))
    if FLAGS.approximate_to_closest_valid_value:
        reconstructed_sequence = approx_each_ans_with_closest_valid_value([reconstructed_sequence])[0]

    pred_differences = np.abs(dishonest_sequence - reconstructed_sequence)
    true_differences = np.abs(resp_honest_seq - dishonest_sequence)
    if pred_n_lies_per_test is not None:
        thr = pred_differences[
            np.argsort(-np.array(pred_differences))[max(0, min(9, int(np.ceil(pred_n_lies_per_test)) - 1))]]
    detected_lies_mask = np.where(pred_differences >= thr, np.ones_like(pred_differences), 0)
    true_lies_mask = np.where(true_differences > 0, np.ones_like(pred_differences), 0)

    if np.sum(detected_lies_mask) > 0:
        prec = np.sum(
            np.where((detected_lies_mask == true_lies_mask) & (true_lies_mask == np.ones_like(true_lies_mask)),
                     np.ones_like(detected_lies_mask), 0)) / np.sum(detected_lies_mask)
    else:
        prec = 0
    if np.sum(true_lies_mask) > 0:
        rec = np.sum(np.where((detected_lies_mask == true_lies_mask) & (true_lies_mask == np.ones_like(true_lies_mask)),
                              np.ones_like(detected_lies_mask), 0)) / np.sum(true_lies_mask)
    else:
        rec = 0

    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    if rec >= 0.9 and np.sum(true_lies_mask) >= 9 and prec < 0.2:
        print()
    return f1, prec, rec, detected_lies_mask, dishonest_sequence, resp_honest_seq


def softmax(x, axis=-1):
    return np.exp(x) / (1e-6 + np.sum(np.exp(x), axis=axis))


def compute_avg_n_lies_per_quiz(honest, lies):
    honest = np.array(honest)
    lies = np.array(lies)
    n_lies_per_quiz = []
    for i in range(len(honest)):
        diff_mask_cnt = np.sum(np.where(honest[i] != lies[i], np.ones_like(honest[i]), 0))
        n_lies_per_quiz.append(diff_mask_cnt / len(lies[i]))
    return np.mean(n_lies_per_quiz)


def compute_n_lies_per_sample_dist(honest, dishonest):
    true_lies_mask = np.where(np.abs(honest - dishonest) > 0, np.ones_like(honest), 0)
    n_lies = np.sum(true_lies_mask, axis=-1)
    import collections
    counter = collections.Counter(n_lies)
    return counter


def run():
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    # evaluate on the dataset at fpath, train on the remaining two datasets
    all_fpaths = ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']
    all_fpaths.remove(FLAGS.fpath)
    fnames_mapper = {'data/BF_CTU.csv': 'C', 'data/BF_V.csv': 'S', 'data/BF_OU.csv': 'H'}

    hdata_train, ldata_train = load_data(all_fpaths)
    hdata_test, ldata_test = load_data([FLAGS.fpath])

    hdata_test = np.array(hdata_test, dtype=np.float)
    hdata_train = np.array(hdata_train, dtype=np.float)
    ldata_test = np.array(ldata_test, dtype=np.float)
    ldata_train = np.array(ldata_train, dtype=np.float)

    compute_n_lies_per_sample_dist(hdata_test, ldata_test)

    if FLAGS.fnorm_strategy == 'STScaler':
        scaler = StandardScaler()
        scaler.fit(hdata_train)
        norm_hdata_train = normalize_scores(hdata_train, scaler)
        norm_ldata_train = normalize_scores(ldata_train, scaler)
        norm_hdata_test = normalize_scores(hdata_test, scaler)
        norm_ldata_test = normalize_scores(ldata_test, scaler)
    else:
        scaler = MinMaxScaler()
        scaler.fit(hdata_train)
        norm_hdata_train = normalize_scores(hdata_train, scaler)
        norm_ldata_train = normalize_scores(ldata_train, scaler)
        norm_hdata_test = normalize_scores(hdata_test, scaler)
        norm_ldata_test = normalize_scores(ldata_test, scaler)
    if not FLAGS.log_perf:
        questions_correlation_heatmap(norm_hdata_test, None, fnames_mapper[FLAGS.fpath])
        questions_correlation_heatmap(norm_hdata_test, norm_ldata_test, fnames_mapper[FLAGS.fpath])
        questions_correlation_heatmap(norm_ldata_test, norm_ldata_test, fnames_mapper[FLAGS.fpath], both_dishonest=True)
        n_questions = len(norm_hdata_train[0])
        create_latex_table_mean_std_by_answ(['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv'],
                                            n_questions)

    tf.reset_default_graph()
    tf.set_random_seed(FLAGS.seed)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config, graph=tf.get_default_graph())
    seq_size = norm_hdata_train.shape[-1]
    m = AutoEncoder(FLAGS.femb_size, FLAGS.hidd_size_enc, FLAGS.hidd_size_sa, seq_size, FLAGS.learning_rate)
    sess.run(m.init_op)
    sess.graph.finalize()

    f1s_all_folds = []
    precs_all_folds = []
    recs_all_folds = []
    all_valid_losses = []

    preds_grouped_by_n_lies_all_folds = {}
    kf = KFold(n_splits=10, random_state=FLAGS.seed, shuffle=True)
    for train_data_indices, test_data_indices in kf.split([i for i in range(len(norm_hdata_train))]):
        train_data_i, valid_data_i = train_test_split(train_data_indices, test_size=0.2, random_state=FLAGS.seed)

        train_data = [norm_hdata_train[i] for i in train_data_i]
        valid_data = [norm_hdata_train[i] for i in valid_data_i]

        best_model_path, best_valid_loss, sess, m = train_and_evaluate(sess, m, np.array(train_data),
                                                                       np.array(valid_data))
        curr_f1, curr_prec, curr_rec, curr_preds_grouped_by_n_lies, detected_lies_masks_dishonest_honest_seqs = \
            lie_detection_eval(best_model_path, sess, m, norm_hdata_test, norm_ldata_test, FLAGS.lie_detection_eval_thr)

        all_valid_losses.append(best_valid_loss)

        f1s_all_folds.append(curr_f1)
        precs_all_folds.append(curr_prec)
        recs_all_folds.append(curr_rec)

        for key in curr_preds_grouped_by_n_lies:
            if key not in preds_grouped_by_n_lies_all_folds.keys():
                preds_grouped_by_n_lies_all_folds[key] = {'prec': [], 'rec': [], 'f1': [], 'mae': []}
            for pkey in preds_grouped_by_n_lies_all_folds[key]:
                preds_grouped_by_n_lies_all_folds[key][pkey].extend(curr_preds_grouped_by_n_lies[key][pkey])

    f1, prec, rec = np.mean(f1s_all_folds), np.mean(precs_all_folds), np.mean(recs_all_folds)

    print('AVG VALID LOSS: {}'.format(np.mean(all_valid_losses)))
    return f1, prec, rec, preds_grouped_by_n_lies_all_folds, np.mean(all_valid_losses)


def create_perfs_charts(preds_grouped_by_n_lies):
    if not os.path.exists('./output/{}'.format(exp_suff)):
        os.makedirs('./output/' + exp_suff)
    x = []
    y_prec = []
    y_rec = []
    y_f1 = []
    y_mae = []
    for n_lies_group, perfs in preds_grouped_by_n_lies.items():
        if n_lies_group > 0:
            print('# lies={}: prec={}, rec={}, f1={}, MAE={}'.format(n_lies_group, np.mean(perfs['prec']),
                                                                     np.mean(perfs['rec']),
                                                                     np.mean(perfs['f1']),
                                                                     np.mean(perfs['mae'])))
            x.append(n_lies_group)
            y_prec.append(np.mean(perfs['prec']))
            y_rec.append(np.mean(perfs['rec']))
            y_f1.append(np.mean(perfs['f1']))
            y_mae.append(np.mean(perfs['mae']))

    groups_sizes = [len(preds_grouped_by_n_lies[n]['prec']) for n in x]
    groups_sizes = np.array(groups_sizes) / sum(groups_sizes)
    groups_sizes = groups_sizes[np.argsort(x)]

    y_prec = np.array(y_prec)[np.argsort(x)]
    y_f1 = np.array(y_f1)[np.argsort(x)]
    y_rec = np.array(y_rec)[np.argsort(x)]

    x = np.array(x)[np.argsort(x)]

    if 10 not in x and 'PTSD' not in FLAGS.fpath:
        x = list(x)
        x.append(10.0)
        x = np.array(x)

        y_prec = list(y_prec)
        y_prec.append(0.0)
        y_prec = np.array(y_prec)

        y_rec = list(y_rec)
        y_rec.append(0.0)
        y_rec = np.array(y_rec)

        y_f1 = list(y_f1)
        y_f1.append(0.0)
        y_f1 = np.array(y_f1)

        groups_sizes = list(groups_sizes)
        groups_sizes.append(0)
        groups_sizes = np.array(groups_sizes)

    fname = FLAGS.fpath.split(r'/')[-1].split('.')[0]

    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})

    plt.figure()
    ax = plt.gca()
    plt.bar(x=x, height=y_prec, label='Precision')
    plt.plot(sorted(list(x)), groups_sizes, color='r', label='Group sizes PDF')
    # plt.title('Mean question-level lie detection precision ({})'.format(fname))
    plt.xlabel('Number of faked answers per sample', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0.5, max(x) + 1)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    leg = ax.legend(prop={'size': 20})
    plt.rc('legend', fontsize=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig('./output/{}/{}_{}_hist_prec.png'.format(exp_suff, fname, exp_suff), bbox_inches='tight',
                pad_inches=0.01)

    plt.figure()
    ax = plt.gca()
    plt.bar(x=x, height=y_rec, label='Recall')
    plt.plot(sorted(list(x)), groups_sizes, color='r', label='Group sizes PDF')
    # plt.title('Mean question-level lie detection recall ({})'.format(fname))
    plt.xlabel('Number of faked answers per sample', fontsize=20)
    plt.ylabel('Recall', fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0.5, max(x) + 1)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    leg = ax.legend(prop={'size': 20})
    plt.rc('legend', fontsize=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig('./output/{}/{}_{}_hist_rec.png'.format(exp_suff, fname, exp_suff), bbox_inches='tight',
                pad_inches=0.01)

    plt.figure()
    ax = plt.gca()
    plt.bar(x=x, height=y_f1, label='F1 Score')
    plt.plot(sorted(list(x)), groups_sizes, color='r', label='Group sizes PDF')
    # plt.title('Mean question-level lie detection f1 ({})'.format(fname))
    plt.xlabel('Number of faked answers per sample', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0.5, max(x) + 1)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    leg = ax.legend(prop={'size': 20})
    plt.rc('legend', fontsize=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig('./output/{}/{}_{}_hist_f1.png'.format(exp_suff, fname, exp_suff), bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    args_values = {}
    for arg in vars(FLAGS):
        args_values[arg] = getattr(FLAGS, arg)
    set_best_hparams()  # overwrite hyperparams with the best found during hparam optimization for each collection
    for arg in vars(FLAGS):
        if not FLAGS.log_perf:
            print(arg, ":", getattr(FLAGS, arg))

    headers_line = '\t'.join([str(k) for k in args_values.keys()]).strip('\t') + '\tf1\tprec\trec'
    hparams_line = '\n' + '\t'.join([str(v) for v in args_values.values()]).strip('\t')
    if not os.path.exists(FLAGS.log_file) and FLAGS.log_perf:
        out = open(FLAGS.log_file, 'w')
        out.write(headers_line)
        out.flush()
    else:
        out = open(FLAGS.log_file, 'a')

    f1, prec, rec, preds_grouped_by_n_lies, valid_loss = run()

    if FLAGS.log_perf:
        perfs_str = '\t{}\t{}\t{}'.format(f1, prec, rec)
        fcntl.flock(out, fcntl.LOCK_EX)
        out.write(hparams_line + perfs_str)
        out.flush()
        fcntl.flock(out, fcntl.LOCK_UN)
        out.close()
    else:
        create_perfs_charts(preds_grouped_by_n_lies)

    print('per-question lie detection F1 score: {:.4f}, '
          'per-question lie detection Prec: {:.4f}, '
          'per-question lie detection Rec: {:.4f}'.format(f1, prec, rec))
    print('{:.4f}\t{:.4f}\t{:.4f}'.format(f1, prec, rec))
    shutil.rmtree(model_ckpt_path)
    print(FLAGS.fpath)
