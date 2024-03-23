import numpy as np


def read_data():
    he_l = np.load('HE_L.npy')
    he_a = np.load('HE_A.npy')
    he_b = np.load('HE_B.npy')

    ihc_l = np.load('IHC_L.npy')
    ihc_a = np.load('IHC_A.npy')
    ihc_b = np.load('IHC_B.npy')


if __name__ == '__main__':
    read_data()
