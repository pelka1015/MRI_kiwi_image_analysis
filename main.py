import nibabel as nib
import skimage.filters as sf
import numpy as np
import cv2
import functions_kiwi as kiwi
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.measure import label
from sklearn.metrics import jaccard_score

# Loading images
img_base = nib.load(r'files\kiwi_bergen.nii.gz').get_fdata()
img_pattern= nib.load(r'files\kiwi_bergen_pattern.nii').get_fdata()


#genetating axis line
centers = []
for i in range(round(img_base.shape[2]*0.3), round(img_base.shape[2]*0.7)):
    try:
        slice_start = img_base[:, :, i]
        slice_8bit = cv2.normalize(slice_start, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


        filtered = cv2.medianBlur(slice_8bit, 5)
        filtered = kiwi.enhance_contrast(filtered)

        threshold = sf.threshold_otsu(filtered)
        binary_image = (filtered  > threshold).astype(np.uint8) * 255

        seed_point_background = (0, 0)
        h, w = binary_image.shape  
        mask = np.zeros((h + 2, w + 2), np.uint8)  
        cv2.floodFill(binary_image, mask, seed_point_background, 0)
        eroded = kiwi.apply_opening_2(binary_image, core_size=8, iterations=1)
        x,y = kiwi.center_detection_2d(eroded)
        centers.append((y,x,i))

    except Exception:
        continue

centroid, direction = kiwi.fit_line_3d(centers)



#generating mid slice for core segmentation
mid = int(round(img_base.shape[2]/2,0))
slice_mid = img_base[:, :, mid]
slic_mid_8bit = cv2.normalize(slice_mid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_mid = clahe.apply(slic_mid_8bit)

def log_gradient(x):
    return 1 - np.log1p(x) / np.log(2)  
gradinet_mid = kiwi.apply_custom_radial_gradient(equalized_mid, kiwi.get_point_on_line_by_z(centroid, direction, mid), log_gradient)

blured  = cv2.medianBlur(gradinet_mid,11)

threshold = sf.threshold_otsu(blured) + 40
binary_image = (blured > threshold).astype(int)
binary_image = (blured> threshold).astype(np.uint8) *255

closed_mid = kiwi.apply_opening(binary_image, kernel_size=(5, 5), iterations=1)

w, h = kiwi.get_object_size(closed_mid)



#core segmentation
core = []
base = []
raw = []
for i in range(img_base.shape[2]):
    slice_start = img_base[:, :, i]
    slice_8bit = cv2.normalize(slice_start, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    slice_8bit_blured = cv2.medianBlur(slice_8bit, 7)



    base_blured = median(slice_8bit, footprint=disk(3))
    threshold_base = sf.threshold_otsu(base_blured) - 5
    binary_base = (base_blured > threshold_base).astype(np.uint8) * 255
    negative_base = cv2.bitwise_not(binary_base)   
    base.append(negative_base)

    threshold_raw = sf.threshold_otsu(slice_8bit) - 5
    binary_raw = (slice_8bit > threshold_base).astype(np.uint8) * 255
    negative_raw = cv2.bitwise_not(binary_raw)   
    raw.append(negative_raw)

    threshold = sf.threshold_otsu(slice_8bit_blured)
    binary_image = (slice_8bit_blured > threshold).astype(np.uint8) * 255
    negative = cv2.bitwise_not(binary_image)

    try:
        point = kiwi.get_point_on_line_by_z(centroid, direction, i)
        mask = kiwi.flood_fill_limited_area(negative, point)
    except:
        mask = np.zeros_like(negative, dtype=np.uint8)

    core.append(mask)

mask_3d = np.stack(core, axis=-1).astype(np.uint8)

x, y = kiwi.get_point_on_line_by_z(centroid, direction, mid)
center_point =(int(round(x)),int(round(y)),mid)

extra_object = kiwi.extract_object_3d(mask_3d, center_point, target_value=1)



#generating 3d raw image
img_3d_raw = np.stack(raw, axis=-1)
img_3d_raw = (img_3d_raw > 0).astype(np.uint8)
x, y = kiwi.get_point_on_line_by_z(centroid, direction, mid)



#core scaling
index = len(core)//2
core_max = core[index]
w2, h2 = kiwi.get_object_size(core_max)

x_ratio = w/w2
y_ratio = h/h2

line = centroid, direction
scaled_mask = kiwi.scale_mask_along_line_3d(extra_object, scale_factors=(x_ratio, y_ratio, 1.7), line=line)
small_scaled_mask = kiwi.scale_mask_along_line_3d(extra_object, scale_factors=(1.2, 1.2, 1.0), line=line)



#cropping base image
base = np.stack(base, axis=-1)
croped_base = kiwi.crop_between_masks(base, small_scaled_mask, scaled_mask)



#finding seed centers
croped_base = (croped_base > 0).astype(np.uint8)
seed_centers = kiwi.find_object_centers(croped_base)
print(f"{len(seed_centers)} centers found.")



#region growing
max_volume = 400
counter = 0
masks = kiwi.region_growing_binary_multi(img_3d_raw, seed_centers, max_volume=max_volume)
for i, mask in enumerate(masks):
    if np.sum(mask) == max_volume:
        counter +=1



#filtration of masks
seeds = kiwi.filter_segmented_masks(masks, min_size=30, max_size=350)
print(f"{len(seeds)} masks after entry filtration.")
print("Objects with max value: ",counter,)



#combining seed masks
z, y, x = croped_base.shape
seeds_mask = kiwi.combined_masks(seeds, (z, y, x))



#background segmentation
background = []
for i in range(img_base.shape[2]):
    slice_start = img_base[:, :, i]
    slice_8bit = cv2.normalize(slice_start, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    slice_8bit_blured = median(slice_8bit, footprint=disk(10))

    threshold_background = sf.threshold_otsu(slice_8bit_blured) - 28
    binary_background = (slice_8bit_blured > threshold_background).astype(np.uint8) * 255
    negative_background = cv2.bitwise_not(binary_background)

    try:
        mask = kiwi.flood_fill_limited_area(negative_background, (10, 10), max_area=99999999999999999999999999999999999999999)

        # If mask is too empty fill it completely
        if np.sum(mask) < 0.20 * mask.size:
            mask = np.ones_like(negative_background, dtype=np.uint8)

    except:
        mask = np.ones_like(negative_background, dtype=np.uint8)

    background.append(mask)



mask_3d = np.stack(background, axis=-1).astype(np.uint8)
background_object = kiwi.extract_object_3d(mask_3d, (0,0,0), target_value=1)



#clearing artfacts from seeds
croped_seeds = kiwi.crop_between_masks(seeds_mask, small_scaled_mask, scaled_mask)

scaled_background_object = kiwi.scale_mask_along_line_3d(background_object, scale_factors=(0.8, 0.8, 0.8), line=line)
croped_seeds = kiwi.subtract_binary_masks(croped_seeds,scaled_background_object)
croped_seeds = kiwi.subtract_binary_masks(croped_seeds,background_object)



#comparing with pattern
labeled_image, num_labels = label(img_pattern,connectivity=2,return_num=True, background=0)
print("Elments in pttern:", num_labels)

labeled_image, num_labels = label(croped_seeds,connectivity=2, return_num=True, background=0)
print("Elements in project:", num_labels)



img_pattern_flat = img_pattern.flatten()
croped_seeds_flat = croped_seeds.flatten()

score = jaccard_score(img_pattern_flat, croped_seeds_flat)
print("Jaccard score:", round(score,2))


#saving results
kiwi.create_line_volume(
    img_shape=img_base.shape,
    centroid=centroid,
    direction=direction,
    output_path=r'generated_files\line_with_points.nii.gz',
    thickness=1,
    point_list=centers,
    point_thickness=1,    
)
print("Saved result to 'line_with_points.nii.gz'")

nib.save(nib.Nifti1Image(scaled_background_object, affine=np.eye(4)), r'generated_files\scaled_background_object_segmentation.nii.gz')
print("Saved result to 'scaled_background_object_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(background_object, affine=np.eye(4)), r'generated_files\background_segmentation.nii.gz')
print("Saved result to 'background_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(croped_seeds, affine=np.eye(4)), r'generated_files\croped_seeds_segmentation.nii.gz')
print("Saved result to 'croped_seeds_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(extra_object, affine=np.eye(4)), r'generated_files\core_segmentation.nii.gz')
print("Saved result to 'core_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(scaled_mask, affine=np.eye(4)), r'generated_files\scaled_core_segmentatio.nii.gz')
print("Saved result to 'scaled_core_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(croped_base, affine=np.eye(4)), r'generated_files\croped_base_segmentation.nii.gz')
print("Saved result to 'croped_base_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(seeds_mask, affine=np.eye(4)), r'generated_files\seed_segmentation.nii.gz')
print("Saved result to 'seed_segmentation.nii.gz'")

nib.save(nib.Nifti1Image(base, affine=np.eye(4)), r'generated_files\base.nii.gz')
print("Saved result to 'base.nii.gz'")
