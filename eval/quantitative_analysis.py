import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_mean(l, a, b):
    return (l + a + b) / 3


def descriptive_analysis(data, tag):
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
    plt.title(f'Histogram of Bhattacharyya Coefficients - {tag}')
    plt.xlabel('Bhattacharyya Coefficients')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def comparison_analysis(before, after):
    mean_before = np.mean(before)
    mean_after = np.mean(after)

    std_before = np.std(before)
    std_after = np.std(after)

    plt.boxplot([before, after], labels=['Before Normalization', 'After Normalization'])
    plt.title('Comparison of Bhattacharyya Coefficients Before and After Normalization')
    plt.ylabel('Bhattacharyya Coefficients')
    plt.show()

    plt.hist(before, bins=30, alpha=0.5, label='Before Normalization')
    plt.hist(after, bins=30, alpha=0.5, label='After Normalization')
    plt.title('Histogram of Bhattacharyya Coefficients')
    plt.xlabel('Bhattacharyya Coefficients')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    print("Descriptive Statistics:")
    print("Before Normalization - Mean:", mean_before, " Std Dev:", std_before)
    print("After Normalization - Mean:", mean_after, " Std Dev:", std_after, "\n")


def statistic_tests(before, after):
    # H0 -  Bhattacharyya after normalization is NOT significantly different from before normalization
    # H1 -  Bhattacharyya after normalization is significantly different from before normalization

    t_statistic, p_value_t = stats.ttest_rel(before, after, alternative='greater')

    print("Paired t-test:")
    print("T-statistic:", t_statistic)
    print("P-value:", p_value_t)

    alpha = 0.05
    if p_value_t < alpha:
        print("Reject the null hypothesis. Results after normalization have improved.")
    else:
        print("Fail to reject the null hypothesis. No significant improvement after normalization.\n")


if __name__ == '__main__':
    plt.close('all')

    """
        Load data
    """

    he_l = np.load('HE_L.npy')
    he_a = np.load('HE_A.npy')
    he_b = np.load('HE_B.npy')

    he_lab = get_mean(he_l, he_a, he_b)

    he_l_norm = np.load('HE_L_norm.npy')
    he_a_norm = np.load('HE_A_norm.npy')
    he_b_norm = np.load('HE_B_norm.npy')

    he_lab_norm = get_mean(he_l_norm, he_a_norm, he_b_norm)

    ihc_l = np.load('IHC_L.npy')
    ihc_a = np.load('IHC_A.npy')
    ihc_b = np.load('IHC_B.npy')

    ihc_lab = get_mean(ihc_l, ihc_a, ihc_b)

    ihc_l_norm = np.load('IHC_L_norm.npy')
    ihc_a_norm = np.load('IHC_A_norm.npy')
    ihc_b_norm = np.load('IHC_B_norm.npy')

    ihc_lab_norm = get_mean(ihc_l_norm, ihc_a_norm, ihc_b_norm)

    """
        Descriptive analysis
    """

    descriptive_analysis(he_l, 'HE L')
    descriptive_analysis(he_l_norm, 'HE L normalized')

    """
        Before / After normalization comparison       
    """

    comparison_analysis(he_l, he_l_norm)

    """
        Statistical tests of significance
    """

    statistic_tests(he_l, he_l_norm)
