from typing import List
import os
import time
import numpy as np
import warnings, torch
from torchvision.transforms import Resize
import copy
import cv2

from .utils import get_device, log_info, bcolors, resize_mask
from .models import Owlv2Wrapper, SAMWrapper, XMemWrapper


class VLM:
    """
    vlm pipeline concanates the three models: owlv2, sam, and xmem for batch frames
    """

    def __init__(
        self,
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path=None,
        resnet_50_path=None,
        device=get_device(required_memory_mb=10240),
        resize_to=(480, 480),
        category_multiplier=100,
        verbose=None,
        verbose_frame_every=10,
        verbose_to_disk: bool = False,
        log_dir: str = None,
        input_batch_size=1,
        enable_xmem=True,
    ) -> None:
        """
        Initializes the VLM (VoxPoser Landmark) object.

        Args:
            owlv2_model_path (str): The file path to the Owlv2 model.
            sam_model_path (str): The file path to the SAM model.
            xmem_model_path (str): The file path to the XMem model.
            resnet_18_path (str, optional): The file path to the ResNet-18 model. Defaults to None.
            resnet_50_path (str, optional): The file path to the ResNet-50 model. Defaults to None.
            device (str, optional): The device to use for computation. Defaults to the current device.
            resize_to (tuple, optional): The size to resize the frames to in two stages, first for the first frame and then for the rest of the frames. All are resized to the shortest side. Defaults to (480, 480).
            category_multiplier (int, optional): The multiplier to scale the SAM results. Defaults to 100.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to None.
            verbose_frame_every (int, optional): The frequency of verbose frame output. Defaults to 10.
            input_batch_size (int, optional): The batch size for input frames. Defaults to 1.
        """
        self.device = device
        self.log_dir = os.path.join(log_dir if log_dir is not None else "./logs", "vlm")
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
        self.enable_xeme = enable_xmem
        self.owlv2_wrapper = Owlv2Wrapper(
            owlv2_model_path,
            self.device,
            verbose_to_disk=verbose_to_disk,
            log_dir=self.log_dir,
        )
        self.sam_wrapper = SAMWrapper(
            sam_model_path,
            self.device,
            verbose_to_disk=verbose_to_disk,
            log_dir=self.log_dir,
        )
        self.xmem_wrappers = []  # to restore the xmem_wrapper for each different input
        for i in range(input_batch_size):
            self.xmem_wrappers.append(
                XMemWrapper(
                    xmem_model_path,
                    device=self.device,
                    resnet_18_path=resnet_18_path,
                    resnet_50_path=resnet_50_path,
                    verbose_frame_every=verbose_frame_every,
                    verbose_to_disk=verbose_to_disk,
                    log_dir=self.log_dir,
                    name=f"{i}",
                )
                if enable_xmem
                else None
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
        self.verbose = verbose
        self.verbose_to_disk = verbose_to_disk
        self.original_frame_shape = None

    def _resize_frame(self, frame, resize_to):
        def resize_by_short_edge(image, target_size):
            """
            按短边等比例缩放图像

            Args:
                image: 输入图像 (numpy array)
                target_size: 目标短边尺寸

            Returns:
                resized_img: 缩放后的图像
            """
            h, w = image.shape[:2]

            # 计算短边缩放比例
            scale = target_size / min(h, w)

            # 计算新的宽高
            new_h = int(h * scale)
            new_w = int(w * scale)

            # 等比例缩放
            resized_img = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            return resized_img

        # resize frame by the shortest side
        frame = torch.from_numpy(frame)
        if resize_to is not None:
            assert isinstance(resize_to, int)
            resize = Resize(resize_to)
            frame = resize(frame)
        return frame.numpy()

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
        """
        Process the first frame of a video sequence.

        Args:
            target_objects (List[str]): List of target object names.
            frame (np.ndarray): The first frame of the video sequence.
            owlv2_threshold (float, optional): Threshold for object detection using OWLv2. Defaults to 0.2.
            sam_threshold (float, optional): Threshold for object segmentation using SAM. Defaults to 0.5.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            release_video_memory (bool, optional): Whether to release video memory after processing. Defaults to True.
            resize_to (tuple, optional): The desired size to resize the frame. Defaults to None.

        Returns:
            List[np.ndarray]: A list of masks representing the first frame processed for each target object.

        Note:
            frame: The frame is represented as a numpy array. shape: (c, h, w) or (b, c, h, w), where b is the batch size.
            The masks are represented as numpy arrays.
        """
        assert self.enable_xeme
        frame = copy.deepcopy(frame)
        verbose = self.verbose if self.verbose is not None else verbose
        resize_to = self.first_frame_resize_to if resize_to is None else resize_to
        if len(frame.shape) == 3:
            frames = np.expand_dims(frame, 0)  # shape: (1, h, w, c)
        else:
            frames = frame
        # record the original frame shape
        self.original_frame_shape = frames.shape
        # resize the frame to the target size by the shortest side
        frames = self._resize_frame(frames, resize_to)
        if self.original_frame_shape != frames.shape:
            warnings.warn(
                f"the frame shape has been changed by resizing, {self.original_frame_shape} -> {frames.shape}, so we will resize the finnal mask to the original shape, which may cause some problems."
            )
        else:
            self.original_frame_shape = None
        assert frames.shape[0] == len(self.xmem_wrappers), (
            f"the input batch size is {len(self.xmem_wrappers)}, but the frame batch size is {frames.shape[0]}"
        )
        ret_value = []
        for i, xmem_wrapper in enumerate(self.xmem_wrappers):
            frame = frames[i, ...]
            owlv2_bboxes, owlv2_scores, owlv2_labels = self.owlv2_wrapper.predict(
                frame,
                target_objects,
                threshold=owlv2_threshold,
                verbose=verbose,
                release_memory=release_video_memory,
                log_prefix=f"camera_{i}",
            )
            if len(owlv2_bboxes) == 0:
                log_info(
                    f"no target objects found in camera {i}, skip.",
                    color=bcolors.WARNING,
                )
                empty_mask = np.zeros(
                    self.original_frame_shape[-2:]
                    if self.original_frame_shape is not None
                    else frame.shape[-2:],
                    dtype=np.uint32,
                )
                ret_value.append(empty_mask)  # add a empty mask (all background)
                # xmem_wrapper.process_first_frame(
                #     frame, total_mask, verbose=verbose, inv_resize_to=self.original_frame_shape
                # )
                continue
            total_mask = np.zeros(
                (frame.shape[1], frame.shape[2])
            )  # 用于存储sam的结果, shape: (h, w)
            # 对每个类别进行循环
            for idx, label in enumerate(set(owlv2_labels)):
                # 获取当前类别的bbox
                current_bboxes = [
                    bbox
                    for bbox, lbl in zip(owlv2_bboxes, owlv2_labels)
                    if lbl == label
                ]
                sam_input_bboxes = [current_bboxes]  # batch x num_bbox x 4
                sam_input_bpoints = [
                    [
                        [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]]
                        for bbox in current_bboxes
                    ]
                ]  # every bbox only has one point: batch x num_bbox x point_per_box x 2
                # print(current_bboxes)
                # print(sam_input_bboxes)
                # print(sam_input_bpoints)
                sam_input_lables = [
                    [[1] * len(current_bboxes)]
                ]  # batch x 1 x num_bbox 其实应该改成 batch x num_bbox x 1 没改好像也没事
                # 调用sam_wrapper.predict
                sam_result = self.sam_wrapper.predict(
                    frame,
                    input_bbox=sam_input_bboxes,
                    input_points=sam_input_bpoints,
                    input_labels=sam_input_lables,
                    threshold=sam_threshold,
                    verbose=verbose,
                    release_memory=release_video_memory,
                    log_prefix=f"camera_{i}_class_{target_objects[label]}",  # 为了区分不同类别的输出
                )
                # 将sam_result加上类别前缀标签
                sam_result = sam_result.astype(np.uint32)
                sam_result[sam_result > 0] += (label + 1) * self.category_multiplier
                total_mask += sam_result
            total_mask = total_mask.astype(np.uint32)
            first_mask = xmem_wrapper.process_first_frame(
                frame,
                total_mask,
                verbose=verbose,
                inv_resize_to=self.original_frame_shape,
            )
            ret_value.append(first_mask)
        return ret_value

    def process_frame(
        self,
        frame: np.ndarray,
        verbose=False,
        release_video_memory=False,
        resize_to=None,
    ):
        assert self.enable_xeme
        verbose = self.verbose if self.verbose is not None else verbose
        resize_to = self.frame_resize_to if resize_to is None else resize_to
        if len(frame.shape) == 3:
            frames = np.expand_dims(frame, 0)
        else:
            frames = frame

        frames = self._resize_frame(frames, resize_to)
        ret_value = []
        for i, xmem_wrapper in enumerate(self.xmem_wrappers):
            frame = frames[i, ...]
            mask = xmem_wrapper.process_frame(
                frame,
                verbose=verbose,
                release_video_memory_every_step=release_video_memory,
                inv_resize_to=self.original_frame_shape,
            )
            ret_value.append(mask)
        return ret_value

    def process_sole_frame(
        self,
        target_objects: List[str],
        frame: np.ndarray,
        owlv2_threshold=0.2,
        sam_threshold=0.5,
        verbose=False,
        release_video_memory=True,
        resize_to=None,
    ):
        """
        Process the first frame of a video sequence.

        Args:
            target_objects (List[str]): List of target object names.
            frame (np.ndarray): The first frame of the video sequence.
            owlv2_threshold (float, optional): Threshold for object detection using OWLv2. Defaults to 0.2.
            sam_threshold (float, optional): Threshold for object segmentation using SAM. Defaults to 0.5.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            release_video_memory (bool, optional): Whether to release video memory after processing. Defaults to True.
            resize_to (tuple, optional): The desired size to resize the frame. Defaults to None.

        Returns:
            List[np.ndarray]: A list of masks representing the first frame processed for each target object.

        Note:
            frame: The frame is represented as a numpy array. shape: (c, h, w) or (b, c, h, w), where b is the batch size.
            The masks are represented as numpy arrays.
        """
        verbose = self.verbose if self.verbose is not None else verbose
        resize_to = self.first_frame_resize_to if resize_to is None else resize_to
        if len(frame.shape) == 3:
            frames = np.expand_dims(frame, 0)  # shape: (1, h, w, c)
        else:
            frames = frame
        # record the original frame shape
        self.original_frame_shape = frames.shape
        # resize the frame to the target size by the shortest side
        frames = self._resize_frame(frames, resize_to)
        if self.original_frame_shape != frames.shape:
            warnings.warn(
                f"the frame shape has been changed by resizing, {self.original_frame_shape} -> {frames.shape}, so we will resize the finnal mask to the original shape, which may cause some problems."
            )
        else:
            self.original_frame_shape = None
        assert frames.shape[0] == len(self.xmem_wrappers), (
            f"the input batch size is {len(self.xmem_wrappers)}, but the frame batch size is {frames.shape[0]}"
        )
        ret_value = []
        for i, xmem_wrapper in enumerate(self.xmem_wrappers):
            frame = frames[i, ...]
            owlv2_bboxes, owlv2_scores, owlv2_labels = self.owlv2_wrapper.predict(
                frame,
                target_objects,
                threshold=owlv2_threshold,
                verbose=verbose,
                release_memory=release_video_memory,
                log_prefix=f"camera_{i}",
            )
            if len(owlv2_bboxes) == 0:
                log_info(
                    f"no target objects found in camera {i}, skip.",
                    color=bcolors.WARNING,
                )
                empty_mask = np.zeros(
                    self.original_frame_shape[-2:]
                    if self.original_frame_shape is not None
                    else frame.shape[-2:],
                    dtype=np.uint32,
                )
                ret_value.append(empty_mask)  # add a empty mask (all background)
                continue
            total_mask = np.zeros(
                (frame.shape[1], frame.shape[2])
            )  # 用于存储sam的结果, shape: (h, w)
            # 对每个类别进行循环
            for idx, label in enumerate(set(owlv2_labels)):
                # 获取当前类别的bbox
                current_bboxes = [
                    bbox
                    for bbox, lbl in zip(owlv2_bboxes, owlv2_labels)
                    if lbl == label
                ]
                sam_input_bboxes = [current_bboxes]  # batch x num_bbox x 4
                sam_input_bpoints = [
                    [
                        [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]]
                        for bbox in current_bboxes
                    ]
                ]  # every bbox only has one point: batch x num_bbox x point_per_box x 2
                # print(current_bboxes)
                # print(sam_input_bboxes)
                # print(sam_input_bpoints)
                sam_input_lables = [
                    [[1] * len(current_bboxes)]
                ]  # batch x 1 x num_bbox 其实应该改成 batch x num_bbox x 1 没改好像也没事
                # 调用sam_wrapper.predict
                sam_result = self.sam_wrapper.predict(
                    frame,
                    input_bbox=sam_input_bboxes,
                    input_points=sam_input_bpoints,
                    input_labels=sam_input_lables,
                    threshold=sam_threshold,
                    verbose=verbose,
                    release_memory=release_video_memory,
                    log_prefix=f"camera_{i}_class_{target_objects[label]}",  # 为了区分不同类别的输出
                )
                # 将sam_result加上类别前缀标签
                sam_result = sam_result.astype(np.uint32)
                sam_result[sam_result > 0] += (label + 1) * self.category_multiplier
                total_mask += sam_result
            total_mask = total_mask.astype(np.uint32)
            prediction = (
                resize_mask(total_mask, self.original_frame_shape[-2:])
                if self.original_frame_shape is not None
                else total_mask
            )
            ret_value.append(prediction)
        return ret_value

    def reset(self):
        pass


# class VLMM:
#     """
#     vlm pipeline concanates the three models: owlv2, sam, and xmem for batch frames

#     Args:
#         owlv2_model_path (str): path to the owlv2 model
#         sam_model_path (str): path to the sam model
#         xmem_model_path (str): path to the xmem model
#     """

#     def __init__(
#         self,
#         owlv2_model_path,
#         sam_model_path,
#         xmem_model_path,
#         resnet_18_path=None,
#         resnet_50_path=None,
#         batch_size=1,
#         device=get_device(),
#     ) -> None:
#         self.device = device
#         self.owlv2_wrapper = Owlv2Wrapper(owlv2_model_path, self.device)
#         self.sam_wrapper = SAMWrapper(sam_model_path, self.device)
#         self.xmem_wrapper = XMemWrapper(
#             xmem_model_path,
#             device=self.device,
#             resnet_18_path=resnet_18_path,
#             resnet_50_path=resnet_50_path,
#         )

#     def process_first_frame(
#         self,
#         target_objects: List[str],
#         frame: np.ndarray,
#         owlv2_threshold=0.2,
#         sam_threshold=0.5,
#         verbose=False,
#         release_video_memory=True,
#     ):
#         owlv2_bboxes, owlv2_scores, owlv2_labels = self.owlv2_wrapper.predict(
#             frame,
#             target_objects,
#             threshold=owlv2_threshold,
#             verbose=verbose,
#             release_memory=release_video_memory,
#         )
#         sam_input_bboxes = [owlv2_bboxes]
#         sam_input_bpoints = [
#             list(
#                 [
#                     [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]]
#                     for bbox in owlv2_bboxes  # every bbox only has one point
#                 ]
#             )
#         ]
#         sam_input_lables = [[[1] * len(owlv2_labels)]]
#         sam_results = self.sam_wrapper.predict(
#             frame,
#             input_bbox=sam_input_bboxes,
#             input_points=sam_input_bpoints,
#             input_labels=sam_input_lables,
#             threshold=sam_threshold,
#             verbose=verbose,
#             release_memory=release_video_memory,
#         )
#         first_mask = self.xmem_wrapper.process_first_frame(
#             frame, sam_results, verbose=verbose
#         )
#         return first_mask

#     def process_frame(
#         self, frame: np.ndarray, verbose=False, release_video_memory=False
#     ):
#         mask = self.xmem_wrapper.process_frame(
#             frame, verbose=verbose, release_video_memory_every_step=release_video_memory
#         )
#         return mask

#     def reset(self):
#         pass
