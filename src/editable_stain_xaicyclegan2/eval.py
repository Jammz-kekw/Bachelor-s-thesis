import torch
import pandas as pd
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image import MemorizationInformedFrechetInceptionDistance, UniversalImageQualityIndex

from editable_stain_xaicyclegan2.model.training_controller import TrainingController
from editable_stain_xaicyclegan2.setup.logging_utils import normalize_image
from editable_stain_xaicyclegan2.setup.settings_module import Settings


if __name__ == '__main__':
    num_exp = input("Enter experiment number or label: ").upper() or "X"
    
    try:
        saved_params = torch.load(f'.mnt/scratch/models/Experiment {num_exp}/final_model_checkpoint.pth')
    except FileNotFoundError:
        saved_params = torch.load(f'.mnt/scratch/models/Experiment {num_exp}/9_model_checkpoint.pth')
    
    training_controller = TrainingController(Settings('settings.cfg'), None, saved_params)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
    psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
    uiqi = UniversalImageQualityIndex().to('cuda')
    mifid_he = MemorizationInformedFrechetInceptionDistance(normalize=False).to('cuda')
    mifid_p63 = MemorizationInformedFrechetInceptionDistance(normalize=False).to('cuda')

    ssim_vals_he = []
    ssim_vals_p63 = []
    psnr_vals_he = []
    psnr_vals_p63 = []
    uiqi_vals_he = []
    uiqi_vals_p63 = []

    testlen = min(len(training_controller.test_he), len(training_controller.test_p63))

    with torch.inference_mode():
        for (real_he, real_p63) in tqdm(zip(training_controller.test_he, training_controller.test_p63), total=testlen):
            fake_he, cycled_he, fake_p63, cycled_p63 = training_controller.eval_step(real_he, real_p63)

            fake_he_norm = normalize_image(fake_he, return_numpy=False, permute=False, squeeze=False).to('cuda')
            fake_p63_norm = normalize_image(fake_p63, return_numpy=False, permute=False, squeeze=False).to('cuda')
            real_he_norm = normalize_image(real_he, return_numpy=False, permute=False, squeeze=False).to('cuda')
            real_p63_norm = normalize_image(real_p63, return_numpy=False, permute=False, squeeze=False).to('cuda')
            cycled_he_norm = normalize_image(cycled_he, return_numpy=False, permute=False, squeeze=False).to('cuda')
            cycled_p63_norm = normalize_image(cycled_p63, return_numpy=False, permute=False, squeeze=False).to('cuda')

            mifid_he.update(real_he_norm, real=True)
            mifid_he.update(fake_he_norm, real=False)

            mifid_p63.update(real_p63_norm, real=True)
            mifid_p63.update(fake_p63_norm, real=False)

            real_he = real_he.cuda()
            real_p63 = real_p63.cuda()

            fake_he_unit = fake_he / 255.0
            fake_p63_unit = fake_p63 / 255.0
            real_he_unit = real_he / 255.0
            real_p63_unit = real_p63 / 255.0
            cycled_he_unit = cycled_he / 255.0
            cycled_p63_unit = cycled_p63 / 255.0

            ssim_vals_he.append(ssim(cycled_he_unit, real_he_unit).cpu().numpy())
            ssim_vals_p63.append(ssim(cycled_p63_unit, real_p63_unit).cpu().numpy())
            psnr_vals_he.append(psnr(cycled_he_unit, real_he_unit).cpu().numpy())
            psnr_vals_p63.append(psnr(cycled_p63_unit, real_p63_unit).cpu().numpy())
            uiqi_vals_he.append(uiqi(cycled_he_unit, real_he_unit).cpu().numpy())
            uiqi_vals_p63.append(uiqi(cycled_p63_unit, real_p63_unit).cpu().numpy())

    # turn all saved metrics into a dataframe
    df = pd.DataFrame({
        'ssim_he': ssim_vals_he,
        'ssim_p63': ssim_vals_p63,
        'psnr_he': psnr_vals_he,
        'psnr_p63': psnr_vals_p63,
        'uiqi_he': uiqi_vals_he,
        'uiqi_p63': uiqi_vals_p63,
        'mifid_he': mifid_he.compute().cpu().numpy(),
        'mifid_p63': mifid_p63.compute().cpu().numpy()
    })
    
    print(df)

    # save dataframe to csv
    df.to_csv(f'.mnt/scratch/models/Experiment {num_exp}/metrics.csv', index=False)
