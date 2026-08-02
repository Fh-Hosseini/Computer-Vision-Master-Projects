import os
import skimage.data
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from selective_search import selective_search
from detection_pipeline  import *

DATA_DIR = "../data/balloon_dataset/test"
MODEL_PATH = './model.joblib'
SCALER_PATH = './scaler.joblib'
THRESH_SIZE = 500

def main():

    # get all images files in the directory and choose one of them to test
    # we can change this idx to test on different images
    image_idx = 0
    files = os.listdir(DATA_DIR)


    # add all images path to a list
    images_path = []
    for file in files:
        if file.lower().endswith(".jpg"):
            images_path.append(os.path.join(DATA_DIR, file))

    image_path = images_path[image_idx]
    image = skimage.io.imread(image_path)
    image_pil = PIL.Image.open(image_path).convert('RGB')

    # perform selective search
    image_label, regions = selective_search(
                            image,
                            scale=500,
                            min_size=20
                        )

    # get the trained model and data scaler from the path
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


    candidates = []
    seen_candidates = set()

    for r in regions:
        # excluding same rectangle
        if tuple(r['rect']) in seen_candidates:
            continue

        seen_candidates.add(tuple(r['rect']))
        
        # excluding regions smaller than THRESH_SIZE pixels
        if r['size'] < THRESH_SIZE:
            continue

        # extracting the features and scale them
        features = extract_features(image_pil, r['rect'])
        if features is None:
            continue
        features = features.reshape(1, -1)
        features = scaler.transform(features)

        # predict the labels using our trained model on extracted features
        probs = model.predict_proba(features)[0]

        if probs[1] >= 0.3:
            candidates.append(r['rect'])

    # Draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1
        )
        ax.add_patch(rect)
    plt.axis('off')


    # saving the image
    if not os.path.isdir('../results/'):
        os.makedirs('../results/')
    fig.savefig('../results/'+image_path.split('/')[-1])
    plt.show()


if __name__ == '__main__':
    main()