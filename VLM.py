from typing import List
import numpy as np
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
