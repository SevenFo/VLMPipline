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

    def process_first_frame(self, target_objects: List[str], frame: np.ndarray):
        owlv2_bboxes, owlv2_scores, owlv2_labels = self.owlv2_wrapper.predict(
            frame, target_objects, threshold=0.2, verbose=True
        )
        owlv2_bpoints = list(
            [
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                for bbox in owlv2_bboxes
            ]
        )
        sam_input_lables = [1] * len(owlv2_labels)
        sam_results = self.sam_wrapper.predict(
            frame,
            input_bbox=owlv2_bboxes,
            input_points=owlv2_bpoints,
            input_labels=sam_input_lables,
            threshold=0.5,
            verbose=True,
        )
        first_mask = self.xmem_wrapper.process_first_frame(
            frame, sam_results, verbose=True
        )

    def process_frame(self, frame: np.ndarray):
        mask = self.xmem_wrapper.process_frame(frame, verbose=True)

    def reset(self):
        pass
