import itertools

import streamlit
import streamlit as st
import torch
import numpy as np
from PIL import Image
from editable_stain_xaicyclegan2.model.dataset import DefaultTransform
from editable_stain_xaicyclegan2.model.mask import get_mask_noise
from editable_stain_xaicyclegan2.setup.logging_utils import normalize_image, load_img_numpy
from editable_stain_xaicyclegan2.model.model import Generator

tf = DefaultTransform()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

gen = Generator(32, 8)
model_dict = torch.load('model_checkpoint.pth')
gen.load_state_dict(model_dict['generator_he_to_p63_state_dict'])
gen = gen.to(device)
gen.eval()


def get_eigen():
    global gen, device

    _eigen = gen.interpretable_conv_2.conv.weight.cpu().detach().numpy().squeeze()
    _eigen /= np.linalg.norm(_eigen, axis=0, keepdims=True)
    _, _eigen_vectors = np.linalg.eig(_eigen.dot(_eigen.T))
    return torch.from_numpy(_eigen_vectors.T).to(device)


def run_model(_img, _eigen_range, _mod_range):
    global gen, device

    # get the eigen vectors
    eigen = get_eigen()
    eigen = eigen[0:_eigen_range, :]
    eigen = torch.tensor(_mod_range).view(_eigen_range, 1).to(device) * eigen

    # prep mask
    mask = get_mask_noise(_img).to(device)

    # get the codes
    img_codes, mask_codes = gen.get_partial_pass(_img, mask)

    # get the new img
    _img = gen.get_modified_rest_pass(_img, img_codes, mask_codes, eigen)

    return normalize_image(_img, channel_reorder=(2, 1, 0))


def prepare_image(path):
    global tf, device

    _img = Image.open(path)

    if _img.mode == 'RGBA':
        _img = _img.convert('RGB')

    _img = tf(_img)
    _img = _img.unsqueeze(0)
    return _img.to(device)


def re_run_model():
    img = prepare_image(st.session_state['uploaded_file'])
    img = run_model(img, st.session_state['eigen_range'], st.session_state['mod_range'])
    st.session_state['img'] = img


def main():
    # put the following two sliders next to each other
    default_file = "test.png"

    eigen_range = st.slider('Top K Eigenvalue Vectors', min_value=1, max_value=16,
                                value=4, step=1, on_change=streamlit.rerun)

    mod_range = [
        st.slider(f'Alpha value {i+1}', min_value=-3.0, max_value=3.0, value=0.0, step=0.1, key=f"mod_{i+1}") for i in range(eigen_range)
    ]

    if 'eigen_range' not in st.session_state:
        st.session_state['eigen_range'] = eigen_range

    if 'mod_range' not in st.session_state:
        st.session_state['mod_range'] = mod_range

    if 'eigen_range' in st.session_state and 'mod_range' in st.session_state:
        if eigen_range != st.session_state['eigen_range'] or any(i != j for i, j in zip(mod_range, st.session_state['mod_range'])):
            st.session_state['eigen_range'] = eigen_range
            st.session_state['mod_range'] = mod_range
            re_run_model()

    # add a file opener for images, and a button to run the model, next to each other
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is None:
        uploaded_file = default_file

    st.session_state['uploaded_file'] = uploaded_file

    # center between columns and stretch to full width
    with st.container():
        run_button = st.button('Run Model', use_container_width=True)

    # if the button is pressed, run the model, use default image if non supplied
    if run_button:
        img = prepare_image(uploaded_file)

        # run the model
        img = run_model(img, eigen_range, mod_range)
        # display the output
        st.image(img, use_column_width=True)
    else:
        # display the input
        if 'img' in st.session_state:
            st.image(st.session_state['img'], use_column_width=True)
        else:

            st.image(load_img_numpy(uploaded_file, channel_reorder=(2, 1, 0)), use_column_width=True)


def test():
    img = prepare_image("test.png")
    _ = run_model(img, 5, [2.1, 3.28, 4.74, -4.52, -1.51])


if __name__ == '__main__':
    main()  # comment this out to generate tiled image
    import matplotlib.pyplot as plt
    # prepare a 5x5 grid of images with seaborn
    img = prepare_image("test.png")
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    # Set column and row labels
    alphas = [x * 2.5 for x in range(-2, 3)]
    col_labels = [f"α = {a}" for a in alphas]
    row_labels = ["r = 1", "r = 2", "r = 3", "r = 4", "r = 5"]
    for i, j in itertools.product(range(5), range(5)):
        out_img = run_model(img, i + 1, [alphas[j]] * (i + 1))

        ax = axes[i, j]
        ax.imshow(out_img)
        ax.axis('off')

        # Set column labels
        if i == 0:  # Set column labels on the top row
            ax.set_title(col_labels[j], fontsize=12)

    # Set row labels
    for i, ax in enumerate(axes[:, 0]):
        ax.annotate(row_labels[i], xy=(-0.02, 0.5), xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    # Adjust layout
    plt.subplots_adjust(left=0.15)  # Adjust this value as needed
    plt.tight_layout()

    plt.savefig("out.png")

