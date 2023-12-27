import kornia
import numpy as np
import torch
from PIL import Image


# Use only if max_length >= 3000 or .mean often, otherwise use RunningMeanStack
# This implementation is faster than RunningMeanStack for large max_length
class RunningMeanStackFast(list):

    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.prev_sum = 0
        self.prev_mean = None
        self.added = []
        self.removed = []
        self.correction = 0

    def append(self, x):
        super().append(x)
        self.added.append(x)
        if len(self) > self.max_length:
            self.removed.append(self.tail)
            self.pop(0)
        else:
            self.prev_sum = sum(self)

    @property
    def mean(self):
        self.correction += 1
        if len(self) == 0:
            return 0
        
        if self.correction % 50 == 0:
            self.correction = 0
            self.prev_mean = sum(self) / len(self)
            return self.prev_mean

        if len(self.added) != 0 and len(self.removed) != 0:
            self.prev_sum += sum(self.added) - sum(self.removed)
            self.prev_mean = self.prev_sum / self.max_length
        else:
            self.prev_mean = self.prev_sum / len(self)
                
        self.added = []
        self.removed = []
        return self.prev_mean
        
    @property
    def head(self):
        return self[-1]

    @property
    def tail(self):
        return self[0]


# running mean for max_length samples using FIFO stack
class RunningMeanStack(list):

    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def append(self, x):
        super().append(x)
        if len(self) > self.max_length:
            self.pop(0)

    @property
    def mean(self):
        if len(self) == 0:
            return 0

        return sum(self) / len(self)
        
    @property
    def head(self):
        return self[-1]

    @property
    def tail(self):
        return self[0]


# return image to normal rgb appearance and value range
def normalize_image(img, return_numpy: bool = True, squeeze: bool = True,
                    permute: bool | tuple[int, int, int, int] = True,
                    channel_reorder: tuple = None):
    img = img.cpu().detach()

    # l_mean: float = 50, l_std: float = 29.59, ab_mean: float = 0, ab_std: float = 74.04
    img[0, 0] = img[0, 0] * 29.59 + 50
    img[0, 1] = img[0, 1] * 74.04
    img[0, 2] = img[0, 2] * 74.04

    # clip values
    img[0, 0] = img[0, 0].clamp(0, 100)
    img[0, 1] = img[0, 1].clamp(-128, 127)
    img[0, 2] = img[0, 2].clamp(-128, 127)

    img = kornia.color.lab_to_rgb(img)
    img *= 255

    if channel_reorder:
        img = img[:, channel_reorder, :, :]

    if permute:
        if isinstance(permute, tuple):
            img = img.permute(*permute)
        else:
            img = img.permute(0, 2, 3, 1)

    if squeeze:
        img = img.squeeze(0)

    if return_numpy:
        return img.numpy().astype('uint8')
    else:
        return img.type(torch.uint8)


def load_img_numpy(img_path, channel_reorder: tuple[int, int, int] | None = None):
    img = Image.open(img_path)
    img = np.array(img)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if channel_reorder:
        img = img[:, :, channel_reorder]

    return img
