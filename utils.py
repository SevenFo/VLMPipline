import torch
from torchvision import transforms
import numpy as np
import open3d as o3d


def get_device():
    if torch.cuda.is_available():
        print("Using GPU")
        device = "cuda"
    else:
        print("CUDA not available. Please connect to a GPU instance if possible.")
        device = "cpu"
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


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (h, w, -1))


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
    return_points = True,
):
    """
    padding can be 0 (None) or 255 (the farthest boundary)
    """
    depth_in_meters = (
        np.array(depth_image, dtype=np.float64) / 255 * (clip_far - clip_near)
    ) + clip_near  # 这里有点问题，实际上应该是 depth_m = near + depth * (far - near)
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
