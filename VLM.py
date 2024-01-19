from typing import List
import numpy as np
from .utils import get_device

from .models import Owlv2Wrapper, SAMWrapper, XMemWrapper


class VLM:
    def __init__(self, owlv2_model_path, sam_model_path, xmem_model_path) -> None:
        self.device = get_device()
        self.owlv2_wrapper = Owlv2Wrapper(owlv2_model_path, self.device)
        self.sam_wrapper = SAMWrapper(sam_model_path, self.device)
        self.xmem_wrapper = XMemWrapper(xmem_model_path, device=self.device)

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

    def process_frame(self, frame: np.ndarray, verbose=False):
        mask = self.xmem_wrapper.process_frame(frame, verbose=verbose)
        return mask

    def reset(self):
        pass
