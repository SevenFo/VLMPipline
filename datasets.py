import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import cv2

from .utils import im_normalization


class VideoDataset(Dataset):
    """
    Dataset for video
    return data:    {   "rgb": rgb image (ndarray) (c,h,w) uint8 0~255,
                        "rgb_tensor": rgb image tensor (torch tensor) (c h w) float32 0~1),
                        "gray": gray image (ndarray) (h,w) uint8 0~255,
                        "info": info dict
                    }
    """

    def __init__(self, video_path, size=480):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.size = size

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if ret:
            info = {}
            data = {}
            info["frame"] = idx
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            min_dim = min(width, height)
            # 计算新的宽度和高度
            new_width = int(width * self.size / min_dim)
            new_height = int(height * self.size / min_dim)
            dim = (new_width, new_height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img_tensor = self.im_transform(img)
            img = img.transpose((2, 0, 1))
            img_gray = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)
            info["shape"] = img.shape[:2]
            print(img.shape)
            data["rgb"] = img
            data["rgb_tensor"] = img_tensor
            data["gray"] = img_gray
            data["info"] = info
            return data
        else:
            raise IndexError("Frame index out of range")

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(
            mask,
            (int(h / min_hw * self.size), int(w / min_hw * self.size)),
            mode="nearest",
        )
