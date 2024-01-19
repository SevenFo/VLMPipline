import cv2
from PIL import Image
import numpy as np
import torch
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

from .utils import trans_bbox
from .datasets import VideoDataset


class Owlv2Wrapper:
    def __init__(self, model_name_or_path, device):
        self.processor = Owlv2Processor.from_pretrained(model_name_or_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name_or_path).to(
            device
        )
        self.device = device
        # change model mode to eval
        self.model.eval()

    def visualization(self, result, image, labels):
        """
        @params: result: list of tuple (box, score, label), box: [x1, y1, x2, y2]
        @params: image: numpy array (h, w, c)
        @params: labels: list of str
        @return: None
        @description: visualize the result
        """
        assert image.shape[-1] == 3, "image should be RGB format"
        plt.figure()
        plt.imshow(image)
        for box, score, label in result:
            box = [round(i, 2) for i in box]
            box = trans_bbox(box)
            print(
                f"Detected {labels[label]} with confidence {round(score, 3)} at location {box}"
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
        plt.show()

    def predict(self, image, texts, threshold=0.5, verbose=False, release_memory=True):
        """
        @params: image: numpy array, only accept one RGB image
        @params: texts: list of list of str or a list of str or single str
        @params: threshold: float
        @return: list of tuple (box, score, label)
        @description: predict the result
        """
        # check image format
        assert image.shape[0] == 3, "image should be RGB format with shape (c, h, w)"
        # preprocess image
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(
            self.device
        )
        # inference
        outputs = self.model(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor(
            [[np.max(image.shape[-2:]), np.max(image.shape[-2:])]]
        ).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )
        # only has one batch
        results = results[0]
        # get boxes score and labels
        boxes = results["boxes"].tolist()  # [(x1, y1, x2, y2), ...)]
        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        all = list(zip(boxes, scores, labels))
        if verbose:
            self.visualization(all, image.transpose(1, 2, 0), labels)
            print(f"Detected {len(all)} objects, boxes, scores, labels are: {all}")
        # score_recorder = {}
        # best = {}
        # for box, score, label in all:
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
        return (boxes, scores, labels)


class SAMWrapper:
    def __init__(self, model_name_or_path, device):
        """
        SAMWrapper constructor.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
        """
        self.model = SamModel.from_pretrained(model_name_or_path).to(device)
        self.processor = SamProcessor.from_pretrained(model_name_or_path)
        self.device = device
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
            color_map (list): The color map.

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
            print(f"label {label} use color {(np.array(color) * 255).astype(np.uint8)}")
            colored_mask[mask == label] = (np.array(color) * 255).astype(np.uint8)
        return colored_mask

    def visualization(self, image, mask, input_bbox=None):
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
            plt.show()
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
            plt.show()

    def predict(
        self,
        image,
        input_points,
        input_bbox=None,
        input_labels=None,
        threshold=0.5,
        verbose=False,
        release_memory=True,
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
        # get masks and scores (as we only get one batch, we only need to get the first one)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
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
                    print(f"label {idx} mask result {m_idx} iou score:{score[m_idx]}")
                    colored_mask = self.colorize_mask(m)
                    plt.subplot(1, 3, m_idx + 1)
                    plt.imshow(colored_mask)
                plt.show()
        # merge all masks to one mask
        best_mask = np.sum(best_masks, axis=0).astype(np.uint8)
        if verbose:
            print(f"best mask:")
            self.visualization(image.transpose(1, 2, 0), best_mask, input_bbox[0])
        if release_memory:
            del inputs
            del outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return best_mask


class XMemWrapper:
    def __init__(
        self,
        model_path,
        video_path=None,
        mask_path=None,
        device="cpu",
        verbose_frame_every=10,
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
        self.load_model()
        self.processor = InferenceCore(self.network, config=self.config)

    def load_model(self):
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

    def process_first_frame(self, first_frame, mask, verbose=False):
        """
        Load model and process the first frame.
        Args:
            first_frame: nparray (C,H,W) uint8 0~255
            mask: nparray (H,W)
        """
        assert len(mask.shape) == 2, "mask dim should be HxW"
        self.num_objects = len(np.unique(mask)) - 1
        print(f"detect {self.num_objects} objects in mask")
        self.processor.set_all_labels(
            range(1, self.num_objects + 1)
        )  # consecutive labels
        self.frame_idx = 0
        with torch.cuda.amp.autocast(enabled=True):
            # convert numpy array to pytorch tensor format (C H W float64 tensor, every pix 0~1)
            # and move to device and normalize
            frame_torch, _ = self.match_image_format(first_frame)
            mask_torch = index_numpy_to_one_hot_torch(mask, self.num_objects + 1).to(
                self.device
            )
            # the background mask is not fed into the model
            prediction = self.processor.step(frame_torch, mask_torch[1:])
            prediction = torch_prob_to_numpy_mask(prediction)
            if verbose:
                visualization = overlay_davis(
                    first_frame.transpose(1, 2, 0), prediction
                )
                display(Image.fromarray(visualization))
            del frame_torch, mask_torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return prediction

    def process_frame(
        self, frame, verbose=False, release_video_memory_every_step=False
    ):
        with torch.cuda.amp.autocast(enabled=True):
            frame_torch, _ = self.match_image_format(frame)
            prediction = self.processor.step(frame_torch)
            prediction = torch_prob_to_numpy_mask(prediction)  # uint8 ndarray
            if verbose and self.frame_idx % self.verbose_frame_every == 0:
                visualization = overlay_davis(frame.transpose(1, 2, 0), prediction)
                display(Image.fromarray(visualization))
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
