from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def get_mean(l, a, b):
    return (l + a + b) / 3


def descriptive_analysis(data, tag, metric):
    """
        Used to get the descriptive analysis of data computed and stored in .npy arrays

        also shows a histogram with pyplot for visual illustration of data distribution

    """

    if metric == 'bha':
        title = "Bhattacharyya vzdialenosť"
    elif metric == 'mt':
        title = "Kullback-Leiberova divergencia"
    else:
        title = "Korelácia"

    mean_value = np.mean(data)
    std_dev = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)
    median_value = np.median(data)
    q1, q2, q3 = np.percentile(data, [25, 50, 75])

    print("Priemer:", mean_value)
    print("Smerodajná odchýlka:", std_dev)
    print("Minimum:", min_value)
    print("Maximum:", max_value)
    print("Medián:", median_value)
    print("Kvartily:")
    print("  Q1:", q1)
    print("  Q2:", q2)
    print("  Q3:", q3, "\n")

    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram {title}  {tag}')
    plt.xlabel(f'{title}')
    plt.ylabel('Frekvencia')
    plt.grid(True)
    plt.show()


def comparison_analysis(before, after, metric):
    """
        Used to compare two sets of data, mostly before and after normalization to see whether data improved

    """

    if metric == 'bha':
        title = "Bhattacharyya vzdialenosť"
    elif metric == 'mt':
        title = "Kullback-Leiberova divergencia"
    else:
        title = "Korelácia"

    mean_before = np.mean(before)
    mean_after = np.mean(after)

    std_before = np.std(before)
    std_after = np.std(after)

    plt.boxplot([before, after], labels=['Pred normalizáciou', 'Po normalizácii'])
    plt.title(f'Porovnanie {title} pred a po normalizácii')
    plt.ylabel(f'{title}')
    plt.text(1.1, mean_before, f'{mean_before:.5f}', color='black', va='center')
    plt.text(2.1, mean_after, f'{mean_after:.5f}', color='black', va='center')
    plt.show()

    plt.hist(before, bins=30, alpha=0.5, label='Pred normalizáciou')
    plt.hist(after, bins=30, alpha=0.5, label='Po normalizácii')
    plt.title(f'Histogram - {title}')
    plt.xlabel(f'{title}')
    plt.ylabel('Frekvencia')
    plt.legend()
    plt.show()

    print(f"Deskriptívna štatistika {title}:")
    print("Pred normalizáciou - Priemer:", mean_before, " Smerodajná odchýlka:", std_before)
    print("After Normalization - Mean:", mean_after, " Smerodajná odchýlka:", std_after, "\n")


def statistic_tests(before, after, metric):
    """
        Runs t-test to prove if there was change in data

        H0 -  Bhattacharyya / Correlation / MT info after normalization is NOT significantly different from before normalization
        H1 -  Bhattacharyya / Correlation / MT info after normalization is significantly different from before normalization

    """

    if metric == ('bha' or 'mt'):
        alt = 'greater'
    else:
        alt = 'less'

    t_statistic, p_value_t = stats.ttest_rel(before, after, alternative=alt)

    print("Párovaný t-test:")
    print("T-štatistika:", t_statistic)
    print("P-hodnota:", p_value_t)

    alpha = 0.05
    if p_value_t < alpha:
        print("Odmietnutie nulovej hypotézy. Výsledky po normalizácii sa zlepšili")
    else:
        print("Neodmietame nulovú hypotézu. Výsledky po normalizácii sa nezlepšili\n\n")


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
    he_norm_mt_info = np.load('4x_mean_new//HE_norm_mt_info.npy')
    he_inter_mt_info = np.load('4x_mean_new//HE_inter_mt_info.npy')

    ihc_gen_mt_info = np.load('4x_mean_new//IHC_gen_mt_info.npy')
    ihc_norm_mt_info = np.load('4x_mean_new//IHC_norm_mt_info.npy')
    ihc_inter_mt_info = np.load('4x_mean_new//IHC_inter_mt_info.npy')

    """
        Run quantitative analysis
    """

    # run_analysis(he_gen_mt_info, he_norm_mt_info, 'HE gen mt info', 'HE norm mt info', 'mt')
    # run_analysis(he_gen_mt_info, he_inter_mt_info, 'HE gen mt info', 'HE inter mt info', 'mt')

    # run_analysis(ihc_gen_mt_info, ihc_norm_mt_info, 'HE gen mt info', 'IHC norm mt info', 'mt')
    # run_analysis(ihc_gen_mt_info, ihc_inter_mt_info, 'IHC gen mt info', 'IHC inter mt info', 'mt')

    # run_analysis(he_lab_bha, he_lab_norm_bha, 'HE generované Bhattacharyya', 'HE posun L histogramu Bhattacharyya', 'bha')
    # run_analysis(he_lab_bha, he_lab_inter_bha, 'HE generované Bhattacharyya', 'HE interpolácia Bhattacharyya', 'bha')

    # run_analysis(ihc_lab_bha, ihc_lab_norm_bha, 'IHC generované Bhattacharyya', 'IHC posun L histogramu Bhattacharyya', 'bha')
    # run_analysis(ihc_lab_bha, ihc_lab_inter_bha, 'IHC generované Bhattacharyya', 'IHC interpolácia Bhattacharyya', 'bha')

    # run_analysis(he_lab_cor, he_lab_norm_cor, 'HE generované Korelácia', 'HE posun L histogramu Korelácia', 'cor')
    # run_analysis(he_lab_cor, he_lab_inter_cor, 'HE generované Korelácia', 'HE interpolácia Korelácia', 'cor')

    # run_analysis(ihc_lab_cor, ihc_lab_norm_cor, 'IHC generované Korelácia', 'IHC posun L histogramu Korelácia', 'cor')
    # run_analysis(ihc_lab_cor, ihc_lab_inter_cor, 'IHC generované Korelácia', 'IHC interpolácia Korelácia', 'cor')
