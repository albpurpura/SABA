import os
import subprocess

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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


def questions_correlation_heatmap(questions_data1, questions_data2, cname, both_dishonest=False):
    plt.figure()
    if both_dishonest:
        hd = 'dis_dis'
    elif questions_data2 is None:
        hd = 'hon_hon'
    else:
        hd = 'hon_dis'

    if questions_data2 is None:
        cmatrix = pd.DataFrame(questions_data1).corr(method='pearson')
        title = 'Honest answers correlation matrix (Pearson correlation coefficient) - {}'.format(cname)
    else:
        cmatrix = np.zeros((len(questions_data1[0]), len(questions_data1[0])))
        for i in range(len(questions_data1[0])):
            for j in range(len(questions_data1[0])):
                cmatrix[i, j] = pd.DataFrame(questions_data1[:, i]).corrwith(pd.DataFrame(questions_data2[:, j]),
                                                                             method='pearson').to_numpy()[0]

        title = 'Honest-Faked answers correlation matrix (Pearson correlation coefficient) - {}'.format(cname)
    if both_dishonest:
        title = 'Faked answers correlation matrix (Pearson correlation coefficient) - {}'.format(cname)
    ax = sns.heatmap(cmatrix, linewidth=0.5, vmin=-1.0, vmax=1.0, square=True, annot=True, fmt='.1f', annot_kws={"size":8})
    print('removed heatmaps title')
    # plt.title(title)
    # plt.show()

    ax.figure.savefig('./output/questions_correlations_{}_{}.png'.format(cname, hd), dpi=600, transparent=True,
                      bbox_inches='tight', pad_inches=0.01)


def create_latex_table_with_mean_std(corrected_data, dishonest_data, honest_data, cname, n_questions):
    file_preamble = '\\documentclass[10pt]{article} \n \\usepackage[usenames]{color} \n \\usepackage{amssymb} \n \\usepackage{amsmath} \n \\usepackage[utf8]{inputenc} \n \\usepackage{multirow} \n  \\usepackage{graphicx} \n \\usepackage{caption} \n \\usepackage{mathtools, nccmath}\\begin{document}'
    file_end = '\\end{document}\n'
    odir = os.path.join(os.getcwd(), 'output/latex/' + cname)
    if not os.path.exists(odir):
        os.makedirs(odir)
    out_path = os.path.join(os.getcwd(),
                            odir + '/latex_mean_stds_differences_with_reconstructed_data_{}.tex'.format(cname))
    out = open(out_path, 'w')
    out.write(file_preamble)
    lines = []
    table_header = ['Measure']
    for i in range(n_questions):
        table_header.append('Q{}'.format(i))
    table_head = r'\begin{table}[h] \centering \resizebox{\linewidth}{!}{ \begin{tabular}{' + ''.join(
        ['l'] * (len(table_header) + 2)) + '}\n & Collection & ' + ' & '.join(table_header).strip(
        ' & ') + r'\\ \hline' + '\n'
    lines.append(table_head)

    means_lines = []
    stds_lines = []
    for (data_name, baseline_data, other_data) in [('corrected-honest', corrected_data, honest_data),
                                                   ('corrected-dishonest', corrected_data, dishonest_data),
                                                   ('dishonest-honest', dishonest_data, honest_data),
                                                   ('dishonest', dishonest_data, dishonest_data),
                                                   ('honest', honest_data, honest_data),
                                                   ('corrected', corrected_data, corrected_data)]:
        data = np.array(baseline_data)
        means_base = np.mean(data, axis=0)
        stds_base = np.std(data, axis=0)
        means_other = np.mean(other_data, axis=0)
        stds_other = np.std(other_data, axis=0)

        assert len(means_base) == data.shape[1]
        assert n_questions == len(means_base)
        if np.mean(means_base) == np.mean(means_other):
            mean_line = data_name + ' answers & ' + cname + ' & Mean & ' + ' & '.join(
                ['{:.2f}'.format(means_base[i]) for i in range(len(means_other))]).strip(
                ' & ') + r'\\ \hline' + '\n'

            std_line = data_name + ' answers & ' + cname + ' & STD & ' + ' & '.join(
                ['{:.2f}'.format(stds_base[i]) for i in range(len(stds_other))]).strip(
                ' & ') + r'\\ \hline' + '\n'
        else:
            mean_line = data_name + ' answers & ' + cname + ' & Mean & ' + ' & '.join(
                ['{:.2f}'.format(means_base[i] - means_other[i]) for i in range(len(means_other))]).strip(
                ' & ') + r'\\ \hline' + '\n'

            std_line = data_name + ' answers & ' + cname + ' & STD & ' + ' & '.join(
                ['{:.2f}'.format(stds_base[i] - stds_other[i]) for i in range(len(stds_other))]).strip(
                ' & ') + r'\\ \hline' + '\n'
        means_lines.append(mean_line)
        stds_lines.append(std_line)

    lines.extend(means_lines)
    lines.extend(stds_lines)

    table_end = r'\end{tabular}} ' \
                + r'\caption{Mean and standard deviation of the answers to each question.} \end{table}'
    lines.append(table_end)
    out.writelines(lines)
    out.write(file_end)
    out.close()
    proc = subprocess.Popen(['cd {}; pdflatex {}'.format(odir, out_path)], shell=True)
    proc.communicate()
    os.system('rm {}'.format(odir + '/*.log'))
    os.system('rm {}'.format(odir + '/*.aux'))


def create_latex_table_mean_std_by_answ(all_fpaths, n_questions):
    fnames_mapper = {'data/BF_CTU.csv': 'C', 'data/BF_V.csv': 'S', 'data/BF_OU.csv': 'HU'}
    file_preamble = '\\documentclass[10pt]{article} \n \\usepackage[usenames]{color} \n \\usepackage{amssymb} \n \\usepackage{amsmath} \n \\usepackage[utf8]{inputenc} \n \\usepackage{multirow} \n  \\usepackage{graphicx} \n \\usepackage{caption} \n \\usepackage{mathtools, nccmath}\\begin{document}'
    file_end = '\\end{document}\n'
    odir = os.path.join(os.getcwd(), 'output/latex')
    out_path = os.path.join(os.getcwd(), 'output/latex/latex_mean_stds_answer_level_data.tex')
    if not os.path.exists('./output/latex'):
        os.makedirs('./output/latex')
    out = open(out_path, 'w')
    out.write(file_preamble)
    for hd_str in ['honest', 'dishonest']:
        lines = []
        table_header = ['Measure']
        for i in range(n_questions):
            table_header.append('Q{}'.format(i))
        table_head = r'\begin{table}[h] \centering \resizebox{\linewidth}{!}{ \begin{tabular}{' + ''.join(
            ['l'] * (len(table_header) + 1)) + '}\n Collection & ' + ' & '.join(table_header).strip(
            ' & ') + r'\\ \hline' + '\n'
        lines.append(table_head)

        for fpath in all_fpaths:
            hdata_test, ldata_test = load_data([fpath])
            hdata_test = np.array(hdata_test, dtype=np.float)
            ldata_test = np.array(ldata_test, dtype=np.float)
            tstat, pvalue = stats.ttest_rel(hdata_test, ldata_test)
            stat_sig = np.where(pvalue < 0.05, np.ones_like(pvalue), 0)
            if hd_str == 'honest':
                data = hdata_test
            else:
                data = ldata_test

            cname = fnames_mapper[fpath]
            data = np.array(data)
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            assert len(means) == data.shape[1]
            assert n_questions == len(means)

            lines.append(
                cname + ' & Mean & ' + ' & '.join(
                    ['{:.2f}{}'.format(means[i], '*' if stat_sig[i] > 0 else '') for i in range(len(means))]).strip(
                    ' & ') + r'\\ \hline' + '\n')
            lines.append(cname + ' & STD & ' + ' & '.join(
                ['{:.2f}{}'.format(stds[i], '*' if stat_sig[i] > 0 else '') for i in range(len(means))]).strip(
                ' & ') + r'\\ \hline' + '\n')

        table_end = r'\end{tabular}} ' \
                    + r'\caption{Mean and standard deviation of the ' + hd_str + r' answers to each question.} \end{table}'
        lines.append(table_end)
        out.writelines(lines)
    out.write(file_end)
    out.close()
    proc = subprocess.Popen(['cd {}; pdflatex {}'.format(odir, out_path)], shell=True)
    proc.communicate()
    os.system('rm {}'.format(odir + '/*.log'))
    os.system('rm {}'.format(odir + '/*.aux'))


def approx_each_ans_with_closest_valid_value(reconstructed_seqs):
    def find_closest_value(curr_v):
        valid_values = [k / 5 for k in range(1, 6)]
        for k in range(len(valid_values)):
            if valid_values[k] - curr_v > 0:
                if k >= 1:
                    return valid_values[k - 1]
                else:
                    return valid_values[k]
        return valid_values[-1]

    approx_seqs = np.zeros_like(reconstructed_seqs)
    for i in range(len(reconstructed_seqs)):
        curr_s = reconstructed_seqs[i]
        for j in range(len(curr_s)):
            approx_curr_v = find_closest_value(curr_s[j])
            approx_seqs[i, j] = approx_curr_v
    return approx_seqs


if __name__ == '__main__':
    create_latex_table_mean_std_by_answ(['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv', 'data/PTSD.csv'], 10)
