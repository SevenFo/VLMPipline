from typing import List
import numpy as np
import cv2

from .utils import get_device
from .models import Owlv2Wrapper, SAMWrapper, XMemWrapper


class VLM:
    """
    vlm pipeline concanates the three models: owlv2, sam, and xmem

    Args:
        owlv2_model_path (str): path to the owlv2 model
        sam_model_path (str): path to the sam model
        xmem_model_path (str): path to the xmem model
    """

    def __init__(
        self,
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path=None,
        resnet_50_path=None,
        device=get_device(),
        resize_to=(480, 480),
        category_multiplier=100,
    ) -> None:
        self.device = device
        self.owlv2_wrapper = Owlv2Wrapper(owlv2_model_path, self.device)
        self.sam_wrapper = SAMWrapper(sam_model_path, self.device)
        self.xmem_wrapper = XMemWrapper(
            xmem_model_path,
            device=self.device,
            resnet_18_path=resnet_18_path,
            resnet_50_path=resnet_50_path,
        )
        if resize_to is not None:
            self.first_frame_resize_to = (
                resize_to[0] if len(resize_to) == 2 else resize_to
            )
            self.frame_resize_to = resize_to[1] if len(resize_to) == 2 else resize_to
        else:
            self.first_frame_resize_to = None
            self.frame_resize_to = None
        self.category_multiplier = (
            category_multiplier  # 用于将sam的结果乘以一个大的常数，然后加到total_mask上
        )

    def _resize_frame(self, frame, resize_to):
        # resize frame by the shortest side
        if resize_to is not None:
            assert isinstance(resize_to, int)
            frame = frame.transpose(1, 2, 0)  # transpose to h,w,c
            h, w = frame.shape[:2]
            if h < w:
                frame = cv2.resize(frame, (resize_to, int(resize_to * h / w)))
            else:
                frame = cv2.resize(frame, (int(resize_to * w / h), resize_to))
            frame = frame.transpose(2, 0, 1)  # transpose back to c,h,w
        return frame

    def process_first_frame(
        self,
        target_objects: List[str],
        frame: np.ndarray,
        owlv2_threshold=0.2,
        sam_threshold=0.5,
        verbose=False,
        release_video_memory=True,
        resize_to=None,
    ):
        resize_to = self.first_frame_resize_to if resize_to is None else resize_to
        frame = self._resize_frame(frame, resize_to)
        owlv2_bboxes, owlv2_scores, owlv2_labels = self.owlv2_wrapper.predict(
            frame,
            target_objects,
            threshold=owlv2_threshold,
            verbose=verbose,
            release_memory=release_video_memory,
        )
        total_mask = np.zeros_like(frame)
        # 对每个类别进行循环
        for i, label in enumerate(set(owlv2_labels)):
            # 获取当前类别的bbox
            current_bboxes = [
                bbox for bbox, lbl in zip(owlv2_bboxes, owlv2_labels) if lbl == label
            ]
            sam_input_bboxes = [current_bboxes]
            sam_input_bpoints = [
                [
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    for bbox in current_bboxes
                ]
            ]  # every bbox only has one point
            sam_input_lables = [[[1] * len(current_bboxes)]]
            # 调用sam_wrapper.predict
            sam_result = self.sam_wrapper.predict(
                frame,
                input_bbox=sam_input_bboxes,
                input_points=sam_input_bpoints,
                input_labels=sam_input_lables,
                threshold=sam_threshold,
                verbose=verbose,
                release_memory=release_video_memory,
            )
            # 将sam_result加上类别前缀标签
            total_mask += sam_result[sam_result > 0] + (
                (label + 1) * self.category_multiplier
            )
        first_mask = self.xmem_wrapper.process_first_frame(
            frame, total_mask, verbose=verbose
        )
        return first_mask

    def process_frame(
        self,
        frame: np.ndarray,
        verbose=False,
        release_video_memory=False,
        resize_to=None,
    ):
        resize_to = self.frame_resize_to if resize_to is None else resize_to
        frame = self._resize_frame(frame, resize_to)
        mask = self.xmem_wrapper.process_frame(
            frame, verbose=verbose, release_video_memory_every_step=release_video_memory
        )
        return mask

    def reset(self):
        pass


class VLMM:
    """
    vlm pipeline concanates the three models: owlv2, sam, and xmem for batch frames

    Args:
        owlv2_model_path (str): path to the owlv2 model
        sam_model_path (str): path to the sam model
        xmem_model_path (str): path to the xmem model
    """

    def __init__(
        self,
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path=None,
        resnet_50_path=None,
        batch_size=1,
        device=get_device(),
    ) -> None:
        self.device = device
        self.owlv2_wrapper = Owlv2Wrapper(owlv2_model_path, self.device)
        self.sam_wrapper = SAMWrapper(sam_model_path, self.device)
        self.xmem_wrapper = XMemWrapper(
            xmem_model_path,
            device=self.device,
            resnet_18_path=resnet_18_path,
            resnet_50_path=resnet_50_path,
        )

    def process_first_frame(
        self,
        target_objects: List[str],
        frame: np.ndarray,
        owlv2_threshold=0.2,
        sam_threshold=0.5,
        verbose=False,
        release_video_memory=True,
    ):
        owlv2_bboxes, owlv2_scores, owlv2_labels = self.owlv2_wrapper.predict(
            frame,
            target_objects,
            threshold=owlv2_threshold,
            verbose=verbose,
            release_memory=release_video_memory,
        )
        sam_input_bboxes = [owlv2_bboxes]
        sam_input_bpoints = [
            list(
                [
                    [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]]
                    for bbox in owlv2_bboxes  # every bbox only has one point
                ]
            )
        ]
        sam_input_lables = [[[1] * len(owlv2_labels)]]
        sam_results = self.sam_wrapper.predict(
            frame,
            input_bbox=sam_input_bboxes,
            input_points=sam_input_bpoints,
            input_labels=sam_input_lables,
            threshold=sam_threshold,
            verbose=verbose,
            release_memory=release_video_memory,
        )
        first_mask = self.xmem_wrapper.process_first_frame(
            frame, sam_results, verbose=verbose
        )
        return first_mask

    def process_frame(
        self, frame: np.ndarray, verbose=False, release_video_memory=False
    ):
        mask = self.xmem_wrapper.process_frame(
            frame, verbose=verbose, release_video_memory_every_step=release_video_memory
        )
        return mask

    def reset(self):
        pass
