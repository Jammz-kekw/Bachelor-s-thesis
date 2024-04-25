from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_mean(l, a, b):
    return (l + a + b) / 3


def descriptive_analysis(data, tag, metric):
    if metric == 'bha':
        title = "Bhattacharyya distance"
    elif metric == 'mt':
        title = "Mutual information"
    else:
        title = "Correlation"

    mean_value = np.mean(data)
    std_dev = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)
    median_value = np.median(data)
    q1, q2, q3 = np.percentile(data, [25, 50, 75])

    print("Mean:", mean_value)
    print("Standard Deviation:", std_dev)
    print("Minimum:", min_value)
    print("Maximum:", max_value)
    print("Median:", median_value)
    print("Quartiles:")
    print("  Q1:\t\t", q1)
    print("  Q2 (Median):", q2)
    print("  Q3:\t\t", q3, "\n")

    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {title}  {tag}')
    plt.xlabel(f'{title}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def comparison_analysis(before, after, metric):
    if metric == 'bha':
        title = "Bhattacharyya distance"
    if metric == 'mt':
        title = "Mutual information"
    else:
        title = "Correlation"

    mean_before = np.mean(before)
    mean_after = np.mean(after)

    std_before = np.std(before)
    std_after = np.std(after)

    plt.boxplot([before, after], labels=['Before Normalization', 'After Normalization'])
    plt.title(f'Comparison of {title} Before and After Normalization')
    plt.ylabel(f'{title}')
    plt.text(1.1, mean_before, f'{mean_before:.5f}', color='black', va='center')
    plt.text(2.1, mean_after, f'{mean_after:.5f}', color='black', va='center')
    plt.show()

    plt.hist(before, bins=30, alpha=0.5, label='Before Normalization')
    plt.hist(after, bins=30, alpha=0.5, label='After Normalization')
    plt.title(f'Histogram of {title}')
    plt.xlabel(f'{title}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    print(f"Descriptive Statistics {title}:")
    print("Before Normalization - Mean:", mean_before, " Std Dev:", std_before)
    print("After Normalization - Mean:", mean_after, " Std Dev:", std_after, "\n")


def statistic_tests(before, after, metric):
    # H0 -  Bhattacharyya / Correlation after normalization is NOT significantly different from before normalization
    # H1 -  Bhattacharyya / Correlation after normalization is significantly different from before normalization

    if metric == 'bha' or 'mt':
        alt = 'greater'
    else:
        alt = 'less'

    t_statistic, p_value_t = stats.ttest_rel(before, after, alternative=alt)

    print("Paired t-test:")
    print("T-statistic:", t_statistic)
    print("P-value:", p_value_t)

    alpha = 0.05
    if p_value_t < alpha:
        print("Reject the null hypothesis. Results after normalization have improved.")
    else:
        print("Fail to reject the null hypothesis. No significant improvement after normalization.\n\n")


def run_analysis(before, after, before_tag, after_tag, metric):
    descriptive_analysis(before, before_tag, metric)
    descriptive_analysis(after, after_tag, metric)

    comparison_analysis(before, after, metric)

    statistic_tests(before, after, metric)


if __name__ == '__main__':
    plt.close()

    """
        Load data Bhattacharyya
    """

    he_l_bha = np.load('4x_mean_new//HE_L_gen_bha.npy')
    he_a_bha = np.load('4x_mean_new//HE_A_gen_bha.npy')
    he_b_bha = np.load('4x_mean_new//HE_B_gen_bha.npy')

    he_lab_bha = get_mean(he_l_bha, he_a_bha, he_b_bha)

    he_l_norm_bha = np.load('4x_mean_new//HE_L_norm_bha.npy')
    he_a_norm_bha = np.load('4x_mean_new//HE_A_norm_bha.npy')
    he_b_norm_bha = np.load('4x_mean_new//HE_B_norm_bha.npy')

    he_lab_norm_bha = get_mean(he_l_norm_bha, he_a_norm_bha, he_b_norm_bha)

    ihc_l_bha = np.load('4x_mean_new//IHC_L_gen_bha.npy')
    ihc_a_bha = np.load('4x_mean_new//IHC_A_gen_bha.npy')
    ihc_b_bha = np.load('4x_mean_new//IHC_B_gen_bha.npy')

    ihc_lab_bha = get_mean(ihc_l_bha, ihc_a_bha, ihc_b_bha)

    ihc_l_norm_bha = np.load('4x_mean_new//IHC_L_norm_bha.npy')
    ihc_a_norm_bha = np.load('4x_mean_new//IHC_A_norm_bha.npy')
    ihc_b_norm_bha = np.load('4x_mean_new//IHC_B_norm_bha.npy')

    ihc_lab_norm_bha = get_mean(ihc_l_norm_bha, ihc_a_norm_bha, ihc_b_norm_bha)

    he_l_inter_bha = np.load('4x_mean_new//HE_L_inter_bha.npy')
    he_a_inter_bha = np.load('4x_mean_new//HE_A_inter_bha.npy')
    he_b_inter_bha = np.load('4x_mean_new//HE_B_inter_bha.npy')

    he_lab_inter_bha = get_mean(he_l_inter_bha, he_a_inter_bha, he_b_inter_bha)

    ihc_l_inter_bha = np.load('4x_mean_new//IHC_L_inter_bha.npy')
    ihc_a_inter_bha = np.load('4x_mean_new//IHC_A_inter_bha.npy')
    ihc_b_inter_bha = np.load('4x_mean_new//IHC_B_inter_bha.npy')

    ihc_lab_inter_bha = get_mean(ihc_l_inter_bha, ihc_a_inter_bha, ihc_b_inter_bha)

    """
        Load data Correlation
    """

    he_l_cor = np.load('4x_mean_new//HE_L_gen_cor.npy')
    he_a_cor = np.load('4x_mean_new//HE_A_gen_cor.npy')
    he_b_cor = np.load('4x_mean_new//HE_B_gen_cor.npy')

    he_lab_cor = get_mean(he_l_cor, he_a_cor, he_b_cor)

    he_l_norm_cor = np.load('4x_mean_new//HE_L_norm_cor.npy')
    he_a_norm_cor = np.load('4x_mean_new//HE_A_norm_cor.npy')
    he_b_norm_cor = np.load('4x_mean_new//HE_B_norm_cor.npy')

    he_lab_norm_cor = get_mean(he_l_norm_cor, he_a_norm_cor, he_b_norm_cor)

    ihc_l_cor = np.load('4x_mean_new//IHC_L_gen_cor.npy')
    ihc_a_cor = np.load('4x_mean_new//IHC_A_gen_cor.npy')
    ihc_b_cor = np.load('4x_mean_new//IHC_B_gen_cor.npy')

    ihc_lab_cor = get_mean(ihc_l_cor, ihc_a_cor, ihc_b_cor)

    ihc_l_norm_cor = np.load('4x_mean_new//IHC_L_norm_cor.npy')
    ihc_a_norm_cor = np.load('4x_mean_new//IHC_A_norm_cor.npy')
    ihc_b_norm_cor = np.load('4x_mean_new//IHC_B_norm_cor.npy')

    ihc_lab_norm_cor = get_mean(ihc_l_norm_cor, ihc_a_norm_cor, ihc_b_norm_cor)

    he_l_inter_cor = np.load('4x_mean_new//HE_L_inter_cor.npy')
    he_a_inter_cor = np.load('4x_mean_new//HE_A_inter_cor.npy')
    he_b_inter_cor = np.load('4x_mean_new//HE_B_inter_cor.npy')

    he_lab_inter_cor = get_mean(he_l_inter_cor, he_a_inter_cor, he_b_inter_cor)

    ihc_l_inter_cor = np.load('4x_mean_new//IHC_L_inter_cor.npy')
    ihc_a_inter_cor = np.load('4x_mean_new//IHC_A_inter_cor.npy')
    ihc_b_inter_cor = np.load('4x_mean_new//IHC_B_inter_cor.npy')

    ihc_lab_inter_cor = get_mean(ihc_l_inter_cor, ihc_a_inter_cor, ihc_b_inter_cor)

    """
        Load data mutual information
    """

    he_gen_mt_info = np.load('4x_mean_new//HE_gen_mt_info.npy')
    he_inter_mt_info = np.load('4x_mean_new//HE_inter_mt_info.npy')

    ihc_gen_mt_info = np.load('4x_mean_new//IHC_gen_mt_info.npy')
    ihc_inter_mt_info = np.load('4x_mean_new//IHC_inter_mt_info.npy')

    """
        Run quantitative analysis
    """

    # run_analysis(he_gen_mt_info, he_inter_mt_info, 'HE gen mt info', 'HE inter mt info', 'mt')
    run_analysis(ihc_gen_mt_info, ihc_inter_mt_info, 'IHC gen mt info', 'IHC inter mt info', 'mt')
