import os
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import SamModel, SamProcessor

# import seaborn as sns
import matplotlib.pyplot as plt
from base64 import b64encode
from IPython.display import HTML, Image as IPImage, display

from .XMem.model.network import XMem
from .XMem.inference.inference_core import InferenceCore
from .XMem.dataset.range_transform import im_normalization
from .XMem.inference.interact.interactive_utils import (
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
    overlay_davis,
)

from .utils import (
    trans_bbox,
    get_memory_usage,
    bcolors,
    get_clock_time,
    log_info,
    is_notebook,
    resize_mask,
    resize_rgb,
)
from .datasets import VideoDataset


class Owlv2Wrapper:
    def __init__(
        self, model_name_or_path, device, verbose_to_disk: bool = False, log_dir=None
    ):
        self.processor = Owlv2Processor.from_pretrained(model_name_or_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name_or_path).to(
            device
        )
        self.device = device
        self.verbose_to_disk = verbose_to_disk
        self.log_dir = log_dir
        self.visualization_dir = os.path.join(log_dir, "visualization")
        if os.path.exists(self.visualization_dir) is False:
            os.makedirs(self.visualization_dir)
        # change model mode to eval
        self.model.eval()

    def visualization(self, result, image, labels, log_prefix=""):
        """
        Visualize the result of object detection on an image.

        Args:
            result (list): List of tuples (box, score, label), where box is a list of [x1, y1, x2, y2].
            image (numpy.ndarray): The input image as a numpy array with shape (h, w, c).
            labels (list): List of strings representing the labels for the detected objects.

        Returns:
            None

        Raises:
            AssertionError: If the image is not in RGB format.

        Description:
            This method visualizes the result of object detection on an image. It plots the image and overlays
            bounding boxes around the detected objects, along with their corresponding labels and confidence scores.
        """
        assert image.shape[-1] == 3, "image should be RGB format"
        plt.figure()
        plt.imshow(image)
        for box, score, label in result:
            box = [round(i, 2) for i in box]
            box = trans_bbox(box)
            log_info(
                f"Detected {labels[label]} with confidence {round(score, 3)} at location {[round(i, 2) for i in box]}"
            )
            plt.gca().add_patch(
                plt.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )
        plt.show() if not self.verbose_to_disk else plt.savefig(
            os.path.join(
                self.visualization_dir,
                f"{log_prefix}_owlv2_output_{get_clock_time(milliseconds=True)}.png",
            )
        )

    def predict(
        self,
        image,
        texts,
        threshold=0.5,
        verbose=False,
        release_memory=True,
        log_prefix="",
    ):
        """
        Predicts the result for the given image and texts.

        Args:
            image (numpy array): Only accepts one RGB image.
            texts (list of list of str or list of str or str): The texts to be used for prediction.
            threshold (float): The threshold value for prediction. Defaults to 0.5.
            verbose (bool): If True, displays additional information during prediction. Defaults to False.
            release_memory (bool): If True, releases memory after prediction. Defaults to True.

        Returns:
            list of tuple: A list of tuples containing the predicted boxes, scores, and labels.

        Raises:
            AssertionError: If the image is not in RGB format with shape (c, h, w).

        """
        log_info("#" * 10 + "Owlv2 Model START" + "#" * 10)
        # check image format
        assert (
            image.shape[0] == 3 and len(image.shape) == 3
        ), "image should be RGB format with shape (c, h, w)"
        # preprocess image
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(
            self.device
        )
        # inference
        outputs = self.model(**inputs)
        # show gpu memory usage
        get_memory_usage()
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor(
            [[np.max(image.shape[-2:]), np.max(image.shape[-2:])]]
        ).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )
        ret_value = []
        # we got a batch of image
        for idx, result in enumerate(results):
            # get boxes score and labels
            boxes = result["boxes"].tolist()  # [(x1, y1, x2, y2), ...)]
            scores = result["scores"].tolist()
            labels = result["labels"].tolist()
            _all = list(zip(boxes, scores, labels))
            ret_value.append((boxes, scores, labels))
            if verbose:
                log_info(f"Detect {texts[idx]}")
                self.visualization(
                    _all, image.transpose(1, 2, 0), texts, log_prefix=log_prefix
                )
            # score_recorder = {}
            # best = {}
            # for box, score, label in _all:
            #     if str(label) not in score_recorder.keys():
            #         score_recorder.update({str(label):score})
            #     else:
            #         if score_recorder[str(label)] > score:
            #             continue
            #     score_recorder[str(label)] = score
            #     best.update({str(label):{'score':score,'bbox':box}})
            #     print(score_recorder)
            # best = list([(value['bbox'], value['score'], int(key))for key, value in best.items()])
        if release_memory:
            del inputs
            del outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        log_info("#" * 10 + "Owlv2 Model END" + "#" * 10)
        return ret_value[0]


class SAMWrapper:
    def __init__(
        self, model_name_or_path, device, verbose_to_disk: bool = False, log_dir=None
    ):
        """
        SAMWrapper constructor.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
        """
        self.model = SamModel.from_pretrained(model_name_or_path).to(device)
        self.processor = SamProcessor.from_pretrained(model_name_or_path)
        self.device = device
        self.verbose_to_disk = verbose_to_disk
        self.log_dir = log_dir
        self.visualization_dir = os.path.join(log_dir, "visualization")
        if os.path.exists(self.visualization_dir) is False:
            os.makedirs(self.visualization_dir)
        self.model.eval()

    def generate_color_map(self, n):
        cmap = plt.get_cmap("viridis")  # 'viridis' is one of the built-in colormaps
        colors = [cmap(i) for i in np.linspace(0, 1, n)]
        return colors

    def colorize_mask(self, mask):
        """
        Colorize different regions in the mask.

        Args:
            mask (numpy array): The mask array.

        Returns:
            colored_mask (numpy array): The colored mask array.
        """
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        unique_labels = np.unique(mask)
        color_map = self.generate_color_map(len(unique_labels))
        for label in unique_labels:
            if label == 0:
                continue
            color = color_map[int(label) % len(color_map)][
                :3
            ]  # Convert label to integer
            log_info(
                f"label {label} use color {(np.array(color) * 255).astype(np.uint8)}"
            )
            colored_mask[mask == label] = (np.array(color) * 255).astype(np.uint8)
        return colored_mask

    def visualization(self, image, mask, input_bbox=None, log_prefix=""):
        """
        Visualize the result.

        Args:
            image (numpy array): The input image array.
            input_bbox (list): The input bounding box coordinates.
            mask (numpy array): The mask array.

        Returns:
            None
        """
        assert image.shape[-1] == 3, "image should be RGB format with shape (h, w, c)"
        # colorize mask
        colored_mask = self.colorize_mask(mask)
        # plot image
        plt.figure()
        # mix image and mask
        assert (
            colored_mask.shape == image.shape
        ), f"image and mask should have same shape, while got {image.shape} and {colored_mask.shape}"
        image = cv2.addWeighted(colored_mask, 0.5, image, 0.5, 0)
        plt.imshow(image)
        if input_bbox is None:
            plt.show() if not self.verbose_to_disk else plt.savefig(
                os.path.join(
                    self.visualization_dir,
                    f"{log_prefix}_sam_output_no_bbox_{get_clock_time(milliseconds=True)}.png",
                )
            )
            return
        else:
            # draw bbox
            for box in input_bbox:
                box = [round(i, 2) for i in box]
                box = trans_bbox(box)  # convert to [x1, y1, w, h]
                plt.gca().add_patch(
                    plt.Rectangle(
                        (box[0], box[1]),
                        box[2],
                        box[3],
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
            plt.show() if not self.verbose_to_disk else plt.savefig(
                os.path.join(
                    self.visualization_dir,
                    f"{log_prefix}_sam_output_best_{get_clock_time(milliseconds=True)}.png",
                )
            )

    def predict(
        self,
        image,
        input_points,
        input_bbox=None,
        input_labels=None,
        threshold=0.5,
        verbose=False,
        release_memory=True,
        log_prefix="",
    ):
        """
        Perform prediction on the image.

        Args:
            image (numpy array): The input image, image format is 3, H, W. uint8 0~255
            input_bbox (list): The input bounding box coordinates.
            (batch_size, boxes_per_image, 4)
            input_points (list, optional): The input points. Defaults to None.
            (batch_size, point_batch_size (i.e. how many segmentation masks do we want the model to predict per input point i.e. how many mask (type)), points_per_batch, 2)
            input_labels (list, optional): The input labels. Defaults to None.
            same shape with input_points
                1:  the point is a point that contains the object of interest
                0:  the point is a point that does not contain the object of interest
                -1: the point corresponds to the background
            threshold (float, optional): The threshold value. Defaults to 0.5.
        Returns:
            best_mask: The best mask.
        """
        log_info("#" * 10 + "SAM Model START" + "#" * 10)
        assert image.shape[0] == 3, "image should be RGB format with shape (c, h, w)"
        # preprocess image and prompt, it seems that it provided resize function?
        inputs = self.processor(
            image,
            input_points=input_points,
            input_boxes=input_bbox,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)
        # inference
        outputs = self.model(**inputs)
        # show gpu memory usage
        get_memory_usage()
        # get masks and scores (as we only get one batch, we only need to get the first one)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (num_bbox, 3, H, W)
        scores = outputs.iou_scores.detach().cpu().numpy()[0]
        # select the best mask of every bbox and merge all masks of different bboxes
        best_masks = []
        for idx, score in enumerate(scores):
            max_score_idx = np.argmax(score)
            # mask = np.zeros(masks.shape[-2:], dtype=np.uint8)
            # mask[np.logical_or.reduce(masks[idx].numpy(), axis=0)] = (
            #     idx + 1
            # )  # to merge top 3 mask

            best_masks.append(
                masks[idx, max_score_idx, :, :].numpy().astype(np.uint8)
                * (idx + 1)  # only use best mask
                # mask
            )
            if verbose:
                plt.figure()
                for m_idx, m in enumerate(masks[idx].numpy()):
                    log_info(
                        f"instance {idx} mask result {m_idx} iou score:{round(float(score[m_idx]),3)}"
                    )
                    colored_mask = self.colorize_mask(m)
                    plt.subplot(1, 3, m_idx + 1)
                    plt.imshow(colored_mask)
                plt.show() if not self.verbose_to_disk else plt.savefig(
                    os.path.join(
                        self.visualization_dir,
                        f"{log_prefix}_sam_output_instance_{idx}_{get_clock_time(milliseconds=True)}.png",
                    )
                )
        # merge all masks to one mask
        best_mask = np.sum(best_masks, axis=0).astype(np.uint8)
        if verbose:
            log_info("best mask:")
            self.visualization(
                image.transpose(1, 2, 0), best_mask, input_bbox[0], log_prefix
            )
        if release_memory:
            del inputs
            del outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        log_info("#" * 10 + "SAM Model END" + "#" * 10)
        return best_mask


class XMemWrapper:
    def __init__(
        self,
        model_path,
        video_path=None,
        mask_path=None,
        resnet_18_path=None,
        resnet_50_path=None,
        device="cpu",
        verbose_frame_every=10,
        verbose_to_disk=False,
        log_dir=None,
        name=None,
    ):
        self.model_path = model_path
        self.video_path = video_path
        self.mask_path = mask_path
        self.device = device
        self.network = None
        self.mask = None
        self.num_objects = None
        self.processor = None
        self.config = {
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
            "enable_long_term": True,
            "enable_long_term_count_usage": True,
            "num_prototypes": 256,
            "min_mid_term_frames": 5,
            "max_mid_term_frames": 10,
            "max_long_term_elements": 10000,
        }
        self.verbose_frame_every = verbose_frame_every
        if self.video_path is not None:
            self.video_dataset = VideoDataset(self.video_path)
        else:
            self.video_dataset = None
        self.network = (
            XMem(
                self.config,
                self.model_path,
                resnet_18_path=resnet_18_path,
                resnet_50_path=resnet_50_path,
            )
            .eval()
            .to(self.device)
        )
        self.processor = InferenceCore(self.network, config=self.config)
        self.initilized = False
        self._name = name if name is not None else id(self)
        self._is_nootbook = is_notebook()
        self.verbose_to_disk = verbose_to_disk
        self.log_dir = log_dir
        self.visualization_dir = os.path.join(log_dir, "visualization")
        if os.path.exists(self.visualization_dir) is False:
            os.makedirs(self.visualization_dir)

    # deprecated
    def load_model(self):
        import warnings

        warnings.warn("This function is deprecated.", DeprecationWarning)
        self.network = XMem(self.config, self.model_path).eval().to(self.device)

    def load_mask(self):
        self.mask = np.array(Image.open(self.mask_path))
        self.num_objects = len(np.unique(self.mask)) - 1

    def display_video(self):
        data_url = (
            "data:video/mp4;base64,"
            + b64encode(open(self.video_path, "rb").read()).decode()
        )
        display(
            HTML(
                """
        <video width=400 controls>
              <source src="%s" type="video/mp4">
        </video>
        """
                % data_url
            )
        )

    def display_mask(self):
        display(IPImage(self.mask_path, width=400))

    def match_image_format(self, image):
        frame = torch.from_numpy(image).float().to(self.device) / 255
        frame_norm = im_normalization(frame)
        return frame_norm, frame

    def process_first_frame(self, first_frame, mask, verbose=False, inv_resize_to=None):
        """
        Load model and process the first frame.

        Args:
            first_frame: nparray (C,H,W) uint8 0~255
                The first frame of the video as a numpy array with shape (C, H, W), where C is the number of channels,
                H is the height, and W is the width. The values should be in the range of 0 to 255.
            mask: nparray (H,W)
                The mask for the first frame as a numpy array with shape (H, W), where H is the height and W is the width.
            verbose: bool, optional
                Whether to display verbose output or not. Default is False.
            inv_resize_to: tuple, optional
                The size to which the first frame and visualization should be resized. Should be a tuple of (height, width).
                Default is None, which means no resizing.

        Returns:
            prediction: nparray (H,W)
                The predicted mask for the first frame as a numpy array with shape (H, W), where H is the height and W is the width.
        """
        log_info("#" * 10 + f"XMEM Model [{self._name}] START" + "#" * 10)
        assert (
            len(mask.shape) == 2
        ), f"mask dim should be HxW, got {len(mask.shape)} of {mask.shape}"
        unique_labels = np.unique(mask)
        self.num_objects = len(unique_labels) - 1
        log_info(f"detect {self.num_objects} objects in mask")
        self.processor.set_all_labels(
            range(1, self.num_objects + 1)
        )  # consecutive labels
        # Create a mapping from new labels to original labels
        self.label_mapping = {
            new_label: original_label
            for new_label, original_label in enumerate(unique_labels, start=0)
        }
        log_info(f"label mapping dict:{self.label_mapping}")
        # Create a mapping from original labels to new labels
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        log_info(f"inv label mapping dict:{self.inverse_label_mapping}")

        self.frame_idx = 0
        with torch.cuda.amp.autocast(enabled=True):
            # convert numpy array to pytorch tensor format (C H W float64 tensor, every pix 0~1)
            # and move to device and normalize
            mask = np.vectorize(self.inverse_label_mapping.get)(mask)
            frame_torch, _ = self.match_image_format(first_frame)
            mask_torch = index_numpy_to_one_hot_torch(mask, self.num_objects + 1).to(
                self.device
            )
            # the background mask is not fed into the model
            prediction = self.processor.step(frame_torch, mask_torch[1:])
            # show gpu memory usage
            get_memory_usage()
            prediction = torch_prob_to_numpy_mask(prediction)
            # resize the prediction to the original size
            prediction = (
                resize_mask(prediction, inv_resize_to[-2:])
                if inv_resize_to is not None
                else prediction
            )
            if verbose:
                # reshape first_frame to H W C
                first_frame_disp = (
                    resize_rgb(first_frame, inv_resize_to[-2:])
                    if inv_resize_to is not None
                    else first_frame
                )
                first_frame_disp_hwc = first_frame_disp.transpose(1, 2, 0)
                visualization_by_individual = overlay_davis(
                    first_frame_disp_hwc, prediction
                )
                if self.verbose_to_disk:
                    plt.imsave(
                        os.path.join(
                            self.visualization_dir,
                            f"camera_{self._name}_xmem_output_first_frame_{get_clock_time(milliseconds=True)}.png",
                        ),
                        visualization_by_individual,
                    )
                else:
                    plt.imshow(visualization_by_individual)
                    plt.show()
                if self._is_nootbook:
                    display(Image.fromarray(visualization_by_individual))

            # Map the prediction labels back to the original labels
            prediction = np.vectorize(self.label_mapping.get)(prediction)
            del frame_torch, mask_torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.initilized = True
            log_info("#" * 10 + f"XMEM Model [{self._name}] END" + "#" * 10)
            return prediction

    def process_frame(
        self,
        frame,
        verbose=False,
        release_video_memory_every_step=False,
        inv_resize_to=None,
    ):
        """
        Process one frame.

        Args:
            frame: The input frame to be processed. shape: (C, H, W)
            verbose: Whether to display verbose information during processing.
            release_video_memory_every_step: Whether to release video memory every step.
                It may slow down the process but save memory.
            inv_resize_to: The size to resize the prediction mask to.
                If None, the prediction mask will not be resized.

        Returns:
            The processed prediction mask. shape: (H, W)

        Raises:
            None.
        """
        if not self.initilized:
            log_info(
                f"XMemWrapper[{self._name}] is not initilized, maybe because the first frame is not processed yet or the mask of first frame is empty, so we will return an empty mask",
                color=bcolors.WARNING,
            )
            return np.zeros(
                inv_resize_to[-2:] if inv_resize_to is not None else frame.shape[-2:],
                dtype=np.uint32,
            )  # return an empty mask
        with torch.cuda.amp.autocast(enabled=True):
            frame_torch, _ = self.match_image_format(frame)
            prediction = self.processor.step(frame_torch)
            prediction = torch_prob_to_numpy_mask(prediction)  # uint8 ndarray
            if verbose and self.frame_idx % self.verbose_frame_every == 0:
                # show gpu memory usage
                get_memory_usage()
                visualization_by_individual = overlay_davis(
                    frame.transpose(1, 2, 0),
                    prediction,
                )
                if self.verbose_to_disk:
                    plt.imsave(
                        os.path.join(
                            self.visualization_dir,
                            f"camera_{self._name}_xmem_output_frame_{self.frame_idx}_{get_clock_time(milliseconds=True)}.png",
                        ),
                        visualization_by_individual,
                    )
                else:
                    plt.imshow(visualization_by_individual)
                    plt.show()
                if self._is_nootbook:
                    display(Image.fromarray(visualization_by_individual))
            prediction = (
                resize_mask(prediction, inv_resize_to[-2:])
                if inv_resize_to is not None
                else prediction
            )
            # Map the prediction labels back to the original labels
            prediction = np.vectorize(self.label_mapping.get)(prediction)
            self.frame_idx += 1
            del frame_torch
            if release_video_memory_every_step and self.device == "cuda":
                torch.cuda.empty_cache()
            return prediction

    def reset(self):
        self.processor.clear_memory()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # deprecated
    def process_video(self, frames_to_propagate=200, visualize_every=20):
        # raise deprecated warning
        import warnings

        warnings.warn(
            "This function is deprecated. Please use process_first_frame and process_frame instead.",
            DeprecationWarning,
        )
        return

        self.load_model()
        self.load_mask()
        self.display_video()
        self.display_mask()

        self.processor = InferenceCore(self.network, config=self.config)
        self.processor.set_all_labels(
            range(1, self.num_objects + 1)
        )  # consecutive labels

        with torch.cuda.amp.autocast(enabled=True):
            for i in range(len(self.video_dataset)):
                # load frame-by-frame
                frame = self.video_dataset[i]["rgb"]
                if i > frames_to_propagate:
                    break

                # convert numpy array to pytorch tensor format
                frame_torch, _ = self.match_image_format(frame)
                if i == 0:
                    # initialize with the mask
                    mask_torch = index_numpy_to_one_hot_torch(
                        self.mask, self.num_objects + 1
                    ).to(self.device)
                    # the background mask is not fed into the model
                    prediction = self.processor.step(frame_torch, mask_torch[1:])
                else:
                    # propagate only
                    prediction = self.processor.step(frame_torch)

                # argmax, convert to numpy
                prediction = torch_prob_to_numpy_mask(prediction)

                if i % visualize_every == 0:
                    visualization = overlay_davis(frame.transpose(1, 2, 0), prediction)
                    display(Image.fromarray(visualization))
