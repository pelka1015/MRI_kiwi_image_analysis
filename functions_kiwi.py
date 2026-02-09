import nibabel as nib
import numpy as np
import cv2
from collections import deque
import os
from scipy.ndimage import affine_transform
from scipy.ndimage import distance_transform_edt
#########################################################################___kiwi axis___#####################################################################################################################################
def enhance_contrast(image, low_perc=80, high_perc=20):
    if image.ndim != 2:
        raise ValueError("Image must be 2D (grayscale)")

    p_low, p_high = np.percentile(image, (low_perc, high_perc))
    stretched = np.clip((image - p_low) / (p_high - p_low), 0, 1)

    return (stretched * 255).astype(np.uint8)


def apply_opening_2(image, core_size=3, iterations=1):

    # Circular structuring element
    core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size))

    # first erode
    eroded = cv2.erode(image, core, iterations=iterations)

    # dilate second
    opened = cv2.dilate(eroded, core, iterations=iterations)

    return opened


def center_detection_2d(binary_image):
    # Find contours
    contour, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contour:
        return None  # No contours found

    # Find the largest contour
    contour = max(contour, key=cv2.contourArea)

    # Calculate moments
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None  # Avoid division by zero

    # Calculate centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


def fit_line_3d(points):
    points = np.array(points)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    direction = vh[0]
    return centroid, direction



def get_point_on_line_by_z(centroid, direction, z_query):
    if direction[2] == 0:
        raise ValueError("Direction vector's z-component cannot be zero.")
    t = (z_query - centroid[2]) / direction[2]
    point = centroid + t * direction
    return point[:2]  # Return only x and y coordinates



def create_line_volume(img_shape, centroid, direction, output_path,
                       thickness=1, point_list=None, point_thickness=1, affine=None):

    volume = np.zeros(img_shape, dtype=np.uint8)
    direction = direction / np.linalg.norm(direction)

    # Draw the line
    for z in range(img_shape[2]):
        point = get_point_on_line_by_z(centroid, direction, z)
        x, y = np.round(point[:2]).astype(int)

        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < img_shape[0] and 0 <= yi < img_shape[1]:
                    volume[xi, yi, z] = 1

    # Draw points if provided
    if point_list is not None:
        for p in point_list:
            x, y, z = np.round(p).astype(int)
            for dx in range(-point_thickness, point_thickness + 1):
                for dy in range(-point_thickness, point_thickness + 1):
                    for dz in range(-point_thickness, point_thickness + 1):
                        xi, yi, zi = x + dx, y + dy, z + dz
                        if (0 <= xi < img_shape[0] and 0 <= yi < img_shape[1] and 0 <= zi < img_shape[2]):
                            volume[xi, yi, zi] = 2  # point value

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set default affine if none provided
    if affine is None:
        affine = np.eye(4)

    # save NIfTI file
    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, output_path)

#########################################################################___analysing central part with seeds___##################################################################################################################################
def apply_custom_radial_gradient(image, center, gradient_func):
    height, width = image.shape
    x0, y0 = center

    # creating coordinate grid
    y_indices, x_indices = np.indices((height, width))

    # Calculate distances from the center
    distances = np.sqrt((x_indices - x0) ** 2 + (y_indices - y0) ** 2)

    # Normalize distances to [0, 1]
    max_distance = np.max(distances)
    normalized = distances / max_distance

    # Apply the custom gradient function
    vectorized_func = np.vectorize(gradient_func)
    gradient = vectorized_func(normalized)

    # Clip gradient values to [0, 1]
    gradient = np.clip(gradient, 0, 1)

    # scaling and conversion
    gradient_mask = (gradient * 255).astype(np.uint8)

    # Apply the gradient mask to the image
    result = cv2.multiply(image, gradient_mask, scale=1.0/255.0)

    return result



def apply_opening(image, kernel_size=(5, 5), iterations=1):

    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Erosion
    eroded = cv2.erode(image, kernel, iterations=iterations)

    # Dylation
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)

    return dilated


def get_object_size(binary_mask):
    # Make sure the mask is in uint8 format
    mask = binary_mask.astype(np.uint8)

    # Find contours 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return (0, 0)  # No contours found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box   
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (w, h)
#########################################################################___opertions on core___##################################################################################################################################
def flood_fill_limited_area(image, start, target_value=255, max_area=4000):
    h, w = image.shape
    x, y = int(round(start[0])), int(round(start[1])) 

    if not (0 <= x < h and 0 <= y < w):
        return np.zeros_like(image, dtype=np.uint8)

    if image[x, y] != target_value:
        return np.zeros_like(image, dtype=np.uint8)

    output_mask = np.zeros_like(image, dtype=np.uint8)
    visited = set()
    queue = deque([(x, y)])
    area = 0

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))

        if image[cx, cy] == target_value:
            output_mask[cx, cy] = 1
            area += 1
            if area > max_area:
                return np.zeros_like(image, dtype=np.uint8)

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < h and 0 <= ny < w:
                    queue.append((nx, ny))

    return output_mask




def extract_object_3d(volume, start, target_value=1):
 
    if volume[start] != target_value:
        return np.zeros_like(volume, dtype=np.uint8)

    output_mask = np.zeros_like(volume, dtype=np.uint8)
    queue = deque([start])

    while queue:
        x, y, z = queue.popleft()
        if volume[x, y, z] == target_value and output_mask[x, y, z] == 0:
            output_mask[x, y, z] = 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if abs(dx) + abs(dy) + abs(dz) == 1:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if 0 <= nx < volume.shape[0] and \
                               0 <= ny < volume.shape[1] and \
                               0 <= nz < volume.shape[2]:
                                queue.append((nx, ny, nz))
    return output_mask



def scale_mask_along_line_3d(mask, scale_factors, line):
    if mask.ndim != 3:
        raise ValueError("Mask must be a 3D volume.")
    
    if np.count_nonzero(mask) == 0:
        raise ValueError("Mask is empty.")
    
    centroid, direction = line
    direction = direction / np.linalg.norm(direction)

    # Build local coordinate system
    z_axis = direction
    tmp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrices
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    R_inv = R.T

    # Scaling matrix
    scale_diag = np.diag([1.0 / scale_factors[0], 1.0 / scale_factors[1], 1.0 / scale_factors[2]])
    affine = R @ scale_diag @ R_inv

    # Calculate offset to keep centroid fixed
    offset = centroid - affine @ centroid

    # Apply affine transformation
    transformed = affine_transform(
        input=mask.astype(float),
        matrix=affine,
        offset=offset,
        output_shape=mask.shape,
        order=0
    )

    return (transformed > 0.5).astype(np.uint8)



def crop_between_masks(img_base, extra_object, scaled_mask):

    inner = (extra_object > 0)
    outer = (scaled_mask > 0)

    # Mask of the area between the two masks
    between_mask = np.logical_and(outer, np.logical_not(inner))

    # Create result image
    result = np.zeros_like(img_base)
    result[between_mask] = img_base[between_mask]

    return result

#########################################################################___operations on seeds___##################################################################################################################################


def find_object_centers(volume, target_value=1):
    volume = volume.astype(np.int16)
    shape = volume.shape  # (z, y, x)
    visited = np.zeros_like(volume, dtype=bool)
    centers = []

    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if volume[z, y, x] == target_value and not visited[z, y, x]:
                    coords = []
                    queue = deque([(z, y, x)])
                    visited[z, y, x] = True

                    while queue:
                        zi, yi, xi = queue.popleft()
                        coords.append((zi, yi, xi))
                        for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                            nz, ny, nx = zi + dz, yi + dy, xi + dx
                            if (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]
                                and volume[nz, ny, nx] == target_value and not visited[nz, ny, nx]):
                                visited[nz, ny, nx] = True
                                queue.append((nz, ny, nx))

                    coords_np = np.array(coords)
                    median_coords = tuple(np.median(coords_np, axis=0).astype(int))

                    if volume[median_coords] != target_value:
                        distances = np.linalg.norm(coords_np - median_coords, axis=1)
                        closest = coords_np[np.argmin(distances)]
                        center = tuple(closest.astype(int))
                    else:
                        center = median_coords

                    centers.append(center)

    return centers



def region_growing_binary_multi(volume, seed_points, max_volume=None):
    assert volume.ndim == 3, "Volume must be a 3D array."

    shape_z, shape_y, shape_x = volume.shape
    visited = np.zeros_like(volume, dtype=bool)
    masks = []

    for point in seed_points:
        z, y, x = map(int, point)

        if not (0 <= z < shape_z and 0 <= y < shape_y and 0 <= x < shape_x):
            continue

        if volume[z, y, x] != 1:
            continue

        if visited[z, y, x]:
            continue

        output_mask = np.zeros_like(volume, dtype=np.uint8)
        queue = deque([(z, y, x)])
        visited[z, y, x] = True
        voxel_count = 0

        while queue:
            if max_volume is not None and voxel_count >= max_volume:
                break

            cz, cy, cx = queue.popleft()
            output_mask[cz, cy, cx] = 1
            voxel_count += 1

            for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < shape_z and 0 <= ny < shape_y and 0 <= nx < shape_x:
                    if not visited[nz, ny, nx] and volume[nz, ny, nx] == 1:
                        visited[nz, ny, nx] = True
                        queue.append((nz, ny, nx))

        masks.append(output_mask)

    return masks



def filter_segmented_masks(masks, min_size, max_size):
    filtered = []
    for i, mask in enumerate(masks):
        volume = np.sum(mask)
        if min_size <= volume <= max_size:
            filtered.append(mask)
    return filtered



def combined_masks(masks, shape):
    polaczona_maska = np.zeros(shape, dtype=np.uint8)

    for mask in masks:
        polaczona_maska = np.logical_or(polaczona_maska, mask)

    return polaczona_maska.astype(np.uint8)






def remove_inner_closer_than_outer(mask_inner, mask_outer):
    assert mask_inner.shape == mask_outer.shape, "Masks must have the same shape."

    # Distance from center of object
    dist_inner = distance_transform_edt(mask_inner)
    dist_outer = distance_transform_edt(mask_outer)

    # Create result mask
    condition = dist_inner > dist_outer
    result = np.zeros_like(mask_inner, dtype=np.uint8)
    result[np.logical_and(mask_inner > 0, condition)] = 1

    return result

def subtract_binary_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    if mask1.shape != mask2.shape:
        raise ValueError("Maski muszą mieć ten sam rozmiar.")

    # Subtract mask2 from mask1
    result = np.logical_and(mask1, np.logical_not(mask2)).astype(np.uint8)
    return result
