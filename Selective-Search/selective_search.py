'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###

    # Create segmentation mask using felzenszwalb
    segments = skimage.segmentation.felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)
    segment_mask = segments.reshape(segments.shape[0], segments.shape[1], 1)

    # Concatenate RGB image with segment mask, so the output has one additional channel contains the region label
    img_extended = np.concatenate([im_orig, segment_mask], axis=2)

    return img_extended

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###

    return np.sum(np.minimum(r1["hist_c"], r2["hist_c"]))


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###

    return np.sum(np.minimum(r1["hist_t"], r2["hist_t"]))


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###

    return 1.0 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###

    bb_size = ((max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) *
        (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])))
    return 1.0 - (bb_size - r1["size"] - r2["size"]) / imsize

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = []
    ### YOUR CODE HERE ###

    # For each HSV channel compute histogram
    for i in range(3):
        channel = img[:, i]
        channel_hist, _ = np.histogram(channel, bins=BINS, range=(0, 1))
        hist.extend(channel_hist)

    hist = np.array(hist)

    # L1 normalization
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###

    rgb = img[:, :, :3]

    # Compute LBP for each channel
    for i in range(3):
        channel = rgb[:, :, i]
        ret[:, :, i] = skimage.feature.local_binary_pattern(channel, 8, 1, 'uniform')

    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = []
    ### YOUR CODE HERE ###

    for i in range(3):
        texture_values = img[:, i]
        channel_hist, _ = np.histogram(texture_values, bins=BINS, range=(0, 10))
        hist.extend(channel_hist)

    hist = np.array(hist)

    # L1 Normalize: divide by total number of pixels in the region
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###

    # Get the unique region labels
    labels = img[:, :, 3].astype(int)
    labels = np.unique(labels)

    # Convert from RGB to HSV
    rgb = img[:, :, :3]
    hsv = skimage.color.rgb2hsv(rgb)

    # Calculate the gradient of RGB image
    gradient = calc_texture_gradient(rgb)


    for label in labels:

        # Find all x and y coordinates with tha selected label
        mask = img[:, :, 3] == label
        all_y, all_x = np.where(mask)

        label_count = np.sum(mask)

        min_x = np.min(all_x)
        min_y = np.min(all_y)
        max_x = np.max(all_x)
        max_y = np.max(all_y)

        # Get the selected region with this label
        region = hsv[mask]
        texture_region = gradient[mask]

        # Calculate color and texture histogram
        color_hist = calc_colour_hist(region)
        text_hist = calc_texture_hist(texture_region)

        # Save all extracted value
        R[label] = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'size': label_count,
            'hist_c': color_hist,
            'hist_t': text_hist,
            'labels': [label]
        }

    return R





def extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False


    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    region_ids = list(regions.keys())

    for i, id1 in enumerate(region_ids):
        for id2 in region_ids[i + 1:]:
            r1 = regions[id1]
            r2 = regions[id2]

            # Check if the bounding boxes intersect
            if intersect(r1, r2):
                neighbours.append(((id1, r1), (id2, r2)))

    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE
    rt["size"] = new_size
    rt["labels"] = r1["labels"] + r2["labels"]

    # Merge histograms
    rt["hist_c"] = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size
    rt["hist_t"] = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size

    # Merge bounding boxes
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])


    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        tobe_removed = []
        for key in S:
            if i in key or j in key:
                tobe_removed.append(key)


        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for key in tobe_removed:
            S.pop(key)

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        region_neighbours = []
        for key in tobe_removed:
            if key[0] != i and key[0] != j:
                region_neighbours.append(key[0])
            if key[1] != i and key[1] != j:
                region_neighbours.append(key[1])

        for neighbour in region_neighbours:
            S[(t, neighbour)] = calc_sim(R[t], R[neighbour], imsize)


    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for r_id, r_data in R.items():
        rect = (
            r_data["min_x"],
            r_data["min_y"],
            r_data["max_x"] - r_data["min_x"],
            r_data["max_y"] - r_data["min_y"]
        )
        regions.append({
            "rect": rect,
            "size": r_data["size"],
            "labels": r_data["labels"]
        })


    return image, regions


