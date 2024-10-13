import torch
import datetime
from torchvision import transforms
import numpy as np
from PIL import Image
import open3d as o3d


def is_notebook() -> bool:
    # return False
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f"{curr_time.hour}-{curr_time.minute}-{curr_time.second}-{curr_time.microsecond // 1000}"
    else:
        return f"{curr_time.hour}-{curr_time.minute}-{curr_time.second}"


def log_info(info, color=bcolors.OKGREEN):
    print(f"{color}[VLM INFO|{get_clock_time()}] {info}{bcolors.ENDC}")


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        print("CUDA not available. Please connect to a GPU instance if possible.")
        device = "cpu"
    print(f"Default device: {device}")
    return device


im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def trans_bbox(bbox):
    """
    Transform bbox from [x1,y1,x2,y2] to [x1,y1,w,h]
    """
    a, b, c, d = bbox
    return [a, b, c - a, d - b]


def inv_trans_bbox(bbox):
    """
    Transform bbox from [x1,y1,w,h] to [x1,y1,x2,y2], which used by SAM prompt
    """
    a, b, c, d = bbox
    return [a, b, a + c, b + d]


def free_video_memory(*variables_to_del, device="cuda"):
    """
    Free video memory
    """

    assert device == "cuda", "Only support cuda device"

    # delete variables
    for var in variables_to_del:
        del var
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    # print total available memory and used memory (in bytes)
    print(
        f"total memory (in MB): {t/1024/1024}, reserved memory (in MB): {r/1024/1024}, allocated memory (in MB): {a/1024/1024}"
    )


def get_memory_usage(device="cuda"):
    """
    Get memory usage
    """

    assert device == "cuda", "Only support cuda device"

    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    # print total available memory and used memory (in bytes)
    print(
        f"total memory (in MB): {round(t/1024/1024,1)}, reserved memory (in MB): {round(r/1024/1024,1)}, allocated memory (in MB): {round(a/1024/1024,1)}"
    )


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (h, w, -1))


def resize_mask(mask, size):
    """
    Resize a mask to the specified size.

    Args:
        mask (numpy.ndarray): The input mask. shape (height, width)
        size (tuple): The desired size of the mask. shape (height, width)

    Returns:
        numpy.array: The resized mask.
    """
    # 将 np.uint32 转换为 np.int32 以便 torch 处理
    if mask.dtype == np.uint32:
        mask = mask.astype(np.int32)
    # 将 numpy 数组转换为 torch 张量
    mask = torch.from_numpy(mask).unsqueeze(0)
    # 定义 resize 函数
    resize_func = transforms.Resize(size, interpolation=Image.NEAREST)
    # 调整大小并转换回 numpy 数组
    resized_mask = resize_func(mask).squeeze(0).numpy()
    # 将调整大小后的数组转换回 uint32
    resized_mask = resized_mask.astype(np.uint32)
    return resized_mask


def resize_rgb(rgb, size):
    """
    Resize an RGB image to the specified size.

    Args:
        rgb (numpy.ndarray): The input RGB image. shape (3, height, width)
        size (tuple): The desired size of the RGB image. shape (height, width)

    Returns:
        numpy.array: The resized RGB image.
    """
    rgb = torch.from_numpy(rgb)
    resize_func = transforms.Resize(size, interpolation=Image.BILINEAR)
    return resize_func(rgb).numpy()


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate([pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate([world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo


def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1),
    ).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1),
    ).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1
    )
    return uniform_pixel_coords


def pointcloud_from_depth_and_camera_params(
    depth: np.ndarray, extrinsic_params: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsic_params[:3, 3], 0).T
    R = extrinsic_params[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsic_params = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsic_params)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords


def convert_depth_to_pointcloud(
    depth_image,
    extrinsic_params,
    camera_intrinsics,
    clip_far=3.5,
    clip_near=0.05,
    mask=None,
    return_points=True,
):
    """
    padding can be 0 (None) or 255 (the farthest boundary)
    """
    if depth_image.dtype == np.float32:
        depth_in_meters = (
            np.array(depth_image, dtype=np.float64) * (clip_far - clip_near) + clip_near
        )  # 这里有点问题，实际上应该是 depth_m = near + depth * (far - near)
    elif depth_image.dtype == np.uint8:
        depth_in_meters = (
            (np.array(depth_image, dtype=np.float64) / 255 * (clip_far - clip_near))
            + clip_near
        )  # 这里有点问题，实际上应该是 depth_m = near + depth * (far - near)
    else:
        raise ValueError(f"not support depth_image type:{depth_image.dtype}")
    pcl = pointcloud_from_depth_and_camera_params(
        depth_in_meters, extrinsic_params, camera_intrinsics
    )
    if mask is not None:
        mask = mask.astype(np.uint8) > 0
        pcl = pcl[mask]
    if return_points:
        return pcl
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl.reshape(-1, 3))
    return pcd
