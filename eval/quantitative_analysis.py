import numpy as np
import matplotlib.pyplot as plt


def descriptive_analysis(data):
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
    plt.title('Histogram of Bhattacharyya Coefficients')
    plt.xlabel('Bhattacharyya Coefficients')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def get_mean(l, a, b):
    return (l + a + b) / 3


if __name__ == '__main__':
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

    descriptive_analysis(he_l)
    descriptive_analysis(he_l_norm)

    """
        Before / After normalization comparison       
    """


    """
        Statistical tests of significance
    """


