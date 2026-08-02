import os
import PIL
import joblib
import numpy as np
import json
import skimage.io
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import torch

from pycocotools.cocoeval import COCOeval
from sklearn.preprocessing import StandardScaler

from selective_search import selective_search


# Set constant and parameters and data directories
DATA_DIR = "../data/balloon_dataset"

ANNOTATION_FILE_NAME = "_annotations.coco.json"
PROPOSALS_FILE_NAME = "proposals.json"
FEATURES_FILE_NAME = "features.npz"

THRESH_POS = 0.5
THRESH_NEG = 0.25
THRESH_SIZE = 500

# Set up the pretrained resnet model for feature extraction
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights)
resnet_model.fc = nn.Identity()
resnet_model.eval()

# using the preprocessor to do necessary preprocessing transforms for the resnet model
preprocess = weights.transforms()


def generate_proposals(split_dir):
    """
    This function generates proposals and bounding boxes for all images in directory and save them as json file.

    Args:
    split_dir: The directory to the data split (train, valid or test set)

    Returns:
    proposals_dict: A dictionary that the keys are image file names and values are bounding boxes
    """

    files = os.listdir(split_dir)

    proposals_dict = {}

    # Read all images in the directory and generate proposals for each of them
    for file in files:
        if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png'):
            image_path = os.path.join(split_dir, file)
            image = skimage.io.imread(image_path)

            # perform selective search
            image_label, regions = selective_search(
                image,
                scale=500,
                min_size=20
            )

            # Save the rects of bounding boxes for all regions to the dictionary
            rects = []
            for r in regions:
                x, y, w, h = r['rect']
                rects.append([int(x), int(y), int(w), int(h)])

            proposals_dict[file] = rects

    # Save all proposals in a json file in the same data split directory
    proposal_path = os.path.join(split_dir, PROPOSALS_FILE_NAME)
    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(proposals_dict, f)

    return proposals_dict


def calculate_overlap(box_1, box_2):
    """
    This function calculates the intersection area of two bounding boxes using their rect.

    Args:
        box_1: rect of first box
        box_2: rect of second box

    Returns:
        overlap: a metric that tell us how much is the overlap between two boxes.
    """

    # first we need to find the coordinates of the intersect area of two boxes.
    x1 = max(box_1[0], box_2[0])
    x2 = min(box_1[0] + box_1[2], box_2[0] + box_2[2])

    y1 = max(box_1[1], box_2[1])
    y2 = min(box_1[1] + box_1[3], box_2[1] + box_2[3])

    # now we can calculate the width and length of the intersection area
    intersect_width = x2 - x1
    intersect_length = y2 - y1

    # return 0 if there is no overlap between two boxes.
    if intersect_width <= 0 or intersect_length <= 0:
        return 0

    # now by calculating the overlap area and whole area of two boxes we can compute the overlapping measure.
    box_1_area = box_1[2] * box_1[3]
    box_2_area = box_2[2] * box_2[3]
    intersection_area = intersect_width * intersect_length
    whole_area = box_1_area + box_2_area - intersection_area

    overlap_score = intersection_area / whole_area

    return overlap_score


def separate_pos_neg_boxes(proposals, ground_truths, thresh_pos, thresh_neg):
    """
    Using the calculated overlaps we can label the boxes using positive and negative thresholds.
    If the overlap is above positive threshold, the box will be labeled as positive, which means a balloon class,
    and if the overlap is below negative threshold, the box will be labeled as negative for background class,
    all other boxes between these threshold will be ignored.

    Args:
        proposals: a dictionary that contains generated boxes proposals.
        ground_truths: a dictionary that contains ground truth boxes for ballons.
        thresh_pos: a threshold that used for labeling balloon class.
        thresh_neg: a threshold that used for labeling background class.

    """

    if len(ground_truths) == 0:
        return [], []

    positive_samples = []
    negative_samples = []

    # calculate overlaps of each proposal with gt box
    for proposal_box in proposals:
        overlaps = []

        for gt_box in ground_truths:
            overlap = calculate_overlap(proposal_box, gt_box)
            overlaps.append(overlap)

        max_overlap = np.max(overlaps)

        # labeling the balloon class (or the positive class)
        if max_overlap > thresh_pos:
            positive_samples.append(proposal_box)

        # labeling the background class (or the negative class)
        elif max_overlap < thresh_neg:
            negative_samples.append(proposal_box)

    return positive_samples, negative_samples



def extract_features(image_PIL, proposal_rect):
    """
    Crop the image based on the proposal and extract features from an image using resnet model

    Args:
        image_path: path to the image
        proposal_rect: rect of the proposal that we want to extract features from

    Returns:
         features: a numpy array that contains features extracted from the image
    """

    x, y, w, h = proposal_rect

    # ignoring rects with invalid w or h
    if w <= 0 or h <= 0:
        return None

    # Read the image and crop it using the bounding box
    cropped_image = image_PIL.crop((x, y, x + w, y + h))

    # preprocess the cropped image to match the prerequisites of resnet model
    preprocessed_image = preprocess(cropped_image).unsqueeze(0)

    # get the features from resnet
    with torch.no_grad():
        features = resnet_model(preprocessed_image)
    features = features.squeeze().numpy()

    return features


def preprocessing_pipeline(data_split_dir):
    """
    This function is used to preprocess the data.
    It first read or generates the proposapls for the selected data split.
    Then separate positive and negative samples, and extract features using them.

    Args:
        data_split_dir: it tells us that which split of data we want to preprocess: train, valid or test set

    return:
        X: arrays of extracted features
        y: arrays of labels
    """

    # Check if we generated proposals before, so we can just read them from json file, otherwise generate them.
    proposals_path = os.path.join(data_split_dir, PROPOSALS_FILE_NAME)
    if os.path.exists(proposals_path):
        with open(proposals_path, "r") as f:
            proposals_dict = json.load(f)
    else:
        proposals_dict = generate_proposals(data_split_dir)

    # Read the annotation file of data split to get the ground truth boxes of balloons.
    annotation_path = os.path.join(data_split_dir, ANNOTATION_FILE_NAME)
    annotation_coco = COCO(annotation_path)
    image_ids = annotation_coco.getImgIds()


    # get the positive and negative samples and fill in X and y arrays of features and labels for each of them
    features_path = os.path.join(data_split_dir, FEATURES_FILE_NAME)
    # first we check if we created features before, so we can just use them here to speed up
    if os.path.exists(features_path):
        data = np.load(features_path)
        return data["X"], data["y"]


    X = []
    y = []

    # Go through each image id
    for image_id in image_ids:
        # get image data and its annotations
        img_name = annotation_coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(data_split_dir, img_name)

        annotations_ids = annotation_coco.getAnnIds(imgIds=image_id)
        annotations = annotation_coco.loadAnns(annotations_ids)

        # add all ground truth balloon boxes to a list
        ground_truth_boxes = []
        for ann in annotations:
            ground_truth_boxes.append(ann['bbox'])

        if img_name not in proposals_dict:
            continue

        # using the created proposals, now we create positive and negative samples
        pos_samples, neg_samples = separate_pos_neg_boxes(proposals_dict[img_name], ground_truth_boxes, THRESH_POS, THRESH_NEG)

        if len(pos_samples) > 0:
            neg_samples = neg_samples[:len(pos_samples) * 3]

        # to have more positive data, we can append the ground truth boxes to our positive sample
        # we just do this to increasing our training dataset
        if "train" in data_split_dir.lower():
            for box in ground_truth_boxes:
                box_x, box_y, box_w, box_h = box
                pos_samples.append((int(box_x), int(box_y), int(box_w), int(box_h)))


        # read the image and extract the features and fill in X and y using our positive and negative samples
        image_PIL = PIL.Image.open(image_path).convert('RGB')
        for prop in pos_samples:
            features = extract_features(image_PIL, prop)
            if features is None:
                continue
            X.append(features)
            y.append(1)


        # now for the negative samples
        for prop in neg_samples:
            features = extract_features(image_PIL, prop)
            if features is None:
                continue
            X.append(features)
            y.append(0)


    X = np.array(X)
    y = np.array(y)

    # save extracted features to be able to use them later
    np.savez(features_path, X=X, y=y)

    return X, y


def calculate_map(svm, annotation_coco, data_split_dir, scaler):
    """
    Calculate mAP score

    Args:
        svm: SVM model
        annotation_coco: COCO annotation data
        data_split_dir: data split directory
        scaler: standard scaler

    return:
        mAP: mAP score
    """

    # get all image ids
    image_ids = annotation_coco.getImgIds()
    results = []

    proposals_path = os.path.join(data_split_dir, PROPOSALS_FILE_NAME)
    with open(proposals_path, "r") as f:
        proposals_dict = json.load(f)


    for image_id in image_ids:
        # for each image, get its annotations
        img_name = annotation_coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(data_split_dir, img_name)

        image_PIL = PIL.Image.open(image_path).convert('RGB')
        image = skimage.io.imread(image_path)

        if img_name not in proposals_dict:
            continue

        for rect in proposals_dict[img_name]:
            x, y, w, h = rect

            # excluding tiny regions
            if w * h < THRESH_SIZE:
                continue

            # extracting the features and scale them
            features = extract_features(image_PIL, rect)
            if features is None:
                continue
            features = features.reshape(1, -1)
            features = scaler.transform(features)

            probs = svm.predict_proba(features)[0]

            results.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(probs[1])
            })


    coco_dt = annotation_coco.loadRes(results)
    coco_eval = COCOeval(annotation_coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]



def calculate_mabo(proposals_dict, annotation_coco):
    """
    Calculate MABO score which is the Mean Average Best Overlap

    Args:
        proposals_dict: a dictionary containing our proposals
        annotation_coco: a COCO annotation file

    return:
        mobo: mobo score
    """

    image_ids = annotation_coco.getImgIds()
    abo_list = []

    for image_id in image_ids:
        # get image and its annotaions
        img_name = annotation_coco.loadImgs(image_id)[0]['file_name']
        annotations_ids = annotation_coco.getAnnIds(imgIds=image_id)
        annotations = annotation_coco.loadAnns(annotations_ids)

        # if there isn't any generated proposal for this image or no annotation, we ignore this image
        if img_name not in proposals_dict or len(annotations) == 0:
            continue

        proposals = proposals_dict[img_name]
        overlaps = []

        # for each annotation, we will find the maximum overlap from proposals
        for ann in annotations:
            max_overlap = 0
            for prop in proposals:
                overlap = calculate_overlap(prop, ann['bbox'])
                max_overlap = max(max_overlap, overlap)

            overlaps.append(max_overlap)

        # calculate abo which is the mean of all overlaps
        abo = np.mean(overlaps)
        abo_list.append(abo)

    # calculate the mabo which is the mean of abos for all of our images
    mabo = np.mean(abo_list)
    return mabo


def main():

    # Generate proposals for each image in training set and validation set
    # Then extract features from positive and negative samples
    # We do all of these using our preprocessing pipeline to generate our data for classification

    X_train, y_train = preprocessing_pipeline(os.path.join(DATA_DIR, "train"))
    X_val, y_val = preprocessing_pipeline(os.path.join(DATA_DIR, "valid"))
    X_test, y_test = preprocessing_pipeline(os.path.join(DATA_DIR, "test"))

    print("X train shape:", X_train.shape)
    print("y train shape:", y_train.shape)


    # Before classifying the data, we first need to scale them using standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    # To get the best result, we can use a search to find the best hyperparameter C for our classification model
    param_grid = {'C': [0.1, 1, 10, 100]}
    first_svm = SVC(class_weight='balanced', probability=True)
    grid = GridSearchCV(first_svm, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("C after using grid search:", grid.best_params_)

    svm = grid.best_estimator_

    # calculate accuracy on validations
    val_preds = svm.predict(X_val)
    print("Accuracy on validation set:", accuracy_score(y_val, val_preds))


    # Now testing our data and evaluate the model on test set
    print("#" * 50)
    print("Evalucation of test dataset")
    print("#" * 50)

    test_data_path = os.path.join(DATA_DIR, "test")
    test_annotation_path = os.path.join(test_data_path, ANNOTATION_FILE_NAME)
    test_ann = COCO(test_annotation_path)

    # Load the test proposals
    test_prop_path = os.path.join(test_data_path, PROPOSALS_FILE_NAME)
    with open(test_prop_path, "r") as f:
        test_proposals = json.load(f)

    # Evaluate using mabo score and mAP metrics
    mabo = calculate_mabo(test_proposals, test_ann)
    print("MABO Score:", mabo)

    mAP = calculate_map(svm, test_ann, test_data_path, scaler)
    print("mAP Score:", mAP)

    # Save the model and scaler to use them in interface
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(svm, 'model.joblib')


if __name__ == '__main__':
    main()
