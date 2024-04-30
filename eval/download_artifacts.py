""""
    Used to download images from wandb

"""

import wandb

api = wandb.Api()
run = api.run("fiit-bp/editable-stain-xaicyclegan2/9fo6c9sq")

for file in run.files():
    file.download()
