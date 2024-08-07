import multiprocessing
from multiprocessing import Process, Pool
import ctypes, time

import numpy as np

from VLMPipline.VLM import VLM
from .utils import get_device, log_info, bcolors
from utils import timer_decorator


class VLMProcessWrapper(Process):
    multiprocessing.set_start_method("spawn", force=True)

    def __init__(
        self,
        labels,
        frame_shape,
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path=None,
        resnet_50_path=None,
        device=get_device(),
        resize_to=(480, 480),
        category_multiplier=100,
        verbose=None,
        verbose_frame_every=10,
        verbose_to_disk: bool = False,
        log_dir: str = None,
        input_batch_size=1,
    ):
        super(VLMProcessWrapper, self).__init__(
            daemon=True,
        )
        self.labels = labels  # not shared
        self.frame_shape = frame_shape  # not shared
        self.frame_data_size = np.prod(frame_shape).item()  # not shared
        self.bytes_frame = multiprocessing.Array(ctypes.c_uint8, self.frame_data_size)

        self.owlv2_model_path = owlv2_model_path
        self.sam_model_path = sam_model_path
        self.xmem_model_path = xmem_model_path
        self.resnet_18_path = resnet_18_path
        self.resnet_50_path = resnet_50_path
        self.device = device
        self.verbose = verbose
        self.resize_to = resize_to
        self.category_multiplier = category_multiplier
        self.verbose_frame_every = verbose_frame_every
        self.input_batch_size = input_batch_size
        self.verbose_to_disk = verbose_to_disk
        self.log_dir = log_dir

        self.ready_event = multiprocessing.Event()
        self.process_first_frame_event = multiprocessing.Event()
        self.process_frame_event = multiprocessing.Event()
        self.is_processed_first_frame = False

        self.is_running = multiprocessing.Value(ctypes.c_bool, False)
        self.mask_shape = (self.input_batch_size,) + self.frame_shape[-2:]
        self.result = multiprocessing.Array(
            ctypes.c_uint32, np.prod(self.mask_shape).item()
        )

    def start(self):
        self.is_running.value = True
        super(VLMProcessWrapper, self).start()  # start the process

    def get_current_process_id(self):
        return f"{multiprocessing.current_process().ident}:{time.time():.3f}"

    @timer_decorator
    def process_first_frame(self, frame: np.ndarray):
        self.wait_for_ready()
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: set frame to VLM process"+ bcolors.ENDC
        ) if self.verbose else None
        self.set_frame(frame)
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: weakup VLM process to process first frame"+ bcolors.ENDC
        ) if self.verbose else None
        self.process_first_frame_event.set()
        self.wait_for_ready()
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: receive the result of first frame processing"+ bcolors.ENDC
        ) if self.verbose else None
        ret = np.frombuffer(self.result.get_obj(), dtype=np.uint32).reshape(
            self.mask_shape
        )
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: post process frame finished"+ bcolors.ENDC
        ) if self.verbose else None
        return ret

    @timer_decorator
    def process_frame(self, frame: np.ndarray):
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: set frame to VLM process"+ bcolors.ENDC
        ) if self.verbose else None
        self.set_frame(frame)
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: weakup VLM process to process frame"+ bcolors.ENDC
        ) if self.verbose else None
        self.process_frame_event.set()
        self.wait_for_ready()
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: receive the result of frame processing"+ bcolors.ENDC
        ) if self.verbose else None
        ret = np.frombuffer(self.result.get_obj(), dtype=np.uint32).reshape(
            self.mask_shape
        )
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: post process frame finished"+ bcolors.ENDC
        ) if self.verbose else None
        return ret

    def wait_for_ready(self):
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: waiting for VLM process ready"+ bcolors.ENDC
        ) if self.verbose else None
        self.ready_event.wait()
        # if self.ready_event.is_set
        self.ready_event.clear()  # ready for next wait
        print(bcolors.OKCYAN+
            f"[{self.get_current_process_id()}]: VLM process ready"+ bcolors.ENDC
        ) if self.verbose else None

    def reset(self):
        self.is_processed_first_frame = False

    def shutdown(self):
        print(bcolors.WARNING+f"[{self.get_current_process_id()}]: VLM process stopping"+bcolors.ENDC)
        self.is_running.value = False
        self.ready_event.set()
        self.process_frame_event.set()
        self.process_first_frame_event.set()
        self.join()

    def set_frame(self, frame: np.ndarray):
        # TODO: reduce the time consumption, as the frame is copied
        # assert frame.size == self.frame_data_size
        self.bytes_frame[:] = frame.data.tobytes()
        # self.frame_shape = frame.shape

    def run(self):
        print(bcolors.OKCYAN+f"[{self.get_current_process_id()}]: VLM init start"+bcolors.ENDC)
        import torch

        with torch.no_grad():
            self.vlm = VLM(
                self.owlv2_model_path,
                self.sam_model_path,
                self.xmem_model_path,
                self.resnet_18_path,
                self.resnet_50_path,
                device=self.device,
                resize_to=self.resize_to,
                category_multiplier=self.category_multiplier,
                verbose=self.verbose,
                verbose_frame_every=self.verbose_frame_every,
                verbose_to_disk=self.verbose_to_disk,
                log_dir=self.log_dir,
                input_batch_size=self.input_batch_size,
            )
            self.ready_event.set()
            print(bcolors.OKCYAN+
                f"[{self.get_current_process_id()}]: VLM init finished"+bcolors.ENDC
            ) if self.verbose else None
            while self.is_running.value:
                if not self.is_processed_first_frame:
                    print(bcolors.OKCYAN+
                        f"[{self.get_current_process_id()}]: VLM process waiting for process first frame"+bcolors.ENDC
                    ) if self.verbose else None
                    self.process_first_frame_event.wait()
                    print(bcolors.OKCYAN+
                        f"[{self.get_current_process_id()}]: VLM start process first frame"+bcolors.ENDC
                    ) if self.verbose else None
                    frame = np.frombuffer(
                        self.bytes_frame.get_obj(), dtype=np.uint8
                    ).reshape(self.frame_shape)
                    masks = self.vlm.process_first_frame(
                        self.labels, frame, owlv2_threshold=0.15
                    )
                    self.result[:] = np.stack(masks).flatten()
                    print(bcolors.OKCYAN+
                        f"[{self.get_current_process_id()}]: VLM process finished"+bcolors.ENDC
                    ) if self.verbose else None
                    self.is_processed_first_frame = True
                    self.process_first_frame_event.clear()
                    self.ready_event.set()
                else:
                    print(bcolors.OKCYAN+
                        f"[{self.get_current_process_id()}]: VLM process waiting for weakup"+bcolors.ENDC
                    ) if self.verbose else None
                    self.process_frame_event.wait()
                    print(bcolors.OKCYAN+
                        f"[{self.get_current_process_id()}]: VLM start process frame"+bcolors.ENDC
                    ) if self.verbose else None
                    frame = np.frombuffer(
                        self.bytes_frame.get_obj(), dtype=np.uint8
                    ).reshape(self.frame_shape)
                    masks = self.vlm.process_frame(frame, release_video_memory=False)
                    self.result[:] = np.stack(masks).flatten()
                    print(bcolors.OKCYAN+
                        f"[{self.get_current_process_id()}]: VLM process finished"+bcolors.ENDC
                    ) if self.verbose else None
                    self.process_frame_event.clear()
                    self.ready_event.set()
            print(bcolors.WARNING+f"[{self.get_current_process_id()}]: VLM process stopped"+bcolors.ENDC)
