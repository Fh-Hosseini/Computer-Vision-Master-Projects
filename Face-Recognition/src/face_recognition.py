import os
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from face_detector import FaceDetector


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=1, max_distance=1.0, min_prob=0.5):

        self.facenet = FaceNet()

        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    def partial_fit(self, face, label):
        # Add a new aligned face to the gallery with two embeddings for RGB and Gray scale

        # Get color embedding from face
        emb_color = self.facenet.predict(face)

        # Convert to grayscale image to extract embeddings of grayscale image
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Merge gray face with itself to have 3 channels
        gray_face = cv2.merge([gray_face, gray_face, gray_face])
        emb_gray = self.facenet.predict(gray_face)

        # Append embeddings and their labels to the gallery
        self.embeddings = np.vstack([self.embeddings, emb_color[None, :], emb_gray[None, :]])
        self.labels.extend([label, label])


    def predict(self, face, use_distance_limit: bool = True,
    use_prob_limit: bool = True) -> tuple[str, float, float]:

        if len(self.labels) == 0:
            return 'unknown', 0, float('inf')

        # Get color embedding from face
        emb_color = self.facenet.predict(face)

        # Convert to grayscale image to extract embeddings of grayscale image
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Merge gray face with itself to have 3 channels
        gray_face = cv2.merge([gray_face, gray_face, gray_face])
        emb_gray = self.facenet.predict(gray_face)

        # Stack embeddings to compute distances
        embeddings_query = np.vstack([emb_color[None, :], emb_gray[None, :]])

        # Compute pairwise distances of gallery embeddings
        dists = np.linalg.norm(self.embeddings[None, :, :] - embeddings_query[:, None, :], axis=2)
        # average distances over color and grayscale embeddings
        dists_mean = dists.mean(axis=0)

        # Find k nearest neighbors
        idx_knn = np.argsort(dists_mean)[:self.num_neighbours]
        labels_knn = [self.labels[i] for i in idx_knn]

        # Count all the labels
        count_dict = {}
        for label in labels_knn:
            count_dict[label] = count_dict.get(label, 0) + 1


        # Predicted class, which is the label with the maximum count
        pred_label = max(count_dict, key=count_dict.get)
        k_i = count_dict[pred_label]

        # Calculate posterior probability
        prob = k_i / self.num_neighbours

        # Calculate distance to predicted class
        dist_pred = float('inf')
        for i in range(len(idx_knn)):
            if labels_knn[i] == pred_label:
                if dists_mean[idx_knn[i]] < dist_pred:
                    dist_pred = dists_mean[idx_knn[i]]

        if ((use_distance_limit and dist_pred > self.max_distance) or
                (use_prob_limit and prob < self.min_prob)):
                return 'unknown', prob, dist_pred

        return pred_label, prob, dist_pred


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=200):

        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, FaceNet.embedding_dimensionality))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

        self.objectives = None

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )


    def partial_fit(self, face):
        # Get embedding from face and save it
        emb = self.facenet.predict(face)
        self.embeddings = np.vstack([self.embeddings, emb[None, :]])
        return len(self.embeddings) - 1


    # K-means clustering with plotting objectives
    def fit(self, plot=False):

        if len(self.embeddings) < self.num_clusters:
            raise ValueError("There is not enough embeddings to create clusters")

        init_idx = np.random.choice(len(self.embeddings), size=self.num_clusters, replace=False)
        self.cluster_center = self.embeddings[init_idx].copy()

        objective_curve = []
        for i in range(self.max_iter):

            dists = np.linalg.norm(self.embeddings[:, None, :] - self.cluster_center[None, :, :], axis=2)
            membership = np.argmin(dists, axis=1)

            # Compute k-means objective which is the sum of squared distances to assigned centers
            obj = np.sum(np.linalg.norm(self.embeddings - self.cluster_center[membership], axis=1) ** 2)
            objective_curve.append(obj)


            new_centers = np.zeros(self.cluster_center.shape)
            for k in range(self.num_clusters):
                members = self.embeddings[membership == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    # Reinitialize empty cluster
                    idx = np.random.randint(0, len(self.embeddings))
                    new_centers[k] = self.embeddings[idx]

            if np.allclose(new_centers, self.cluster_center):
                print("Converged after {} iterations".format(i))
                break

            self.cluster_center = new_centers


        dists = np.linalg.norm(self.embeddings[:, None, :] - self.cluster_center[None, :, :], axis=2)
        self.cluster_membership = np.argmin(dists, axis=1).tolist()

        self.objectives = objective_curve

        if plot:
            plt.figure()
            plt.plot(objective_curve, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('K-means Objective (Sum of Squared Distances)')
            plt.title('K-means Objective Curve')
            plt.show()

        return self.cluster_center


    def predict(self, face) -> tuple[int, np.ndarray]:
        if len(self.cluster_center) == 0:
            raise ValueError("No cluster centers calculated")

        # Get embedding from face
        emb = self.facenet.predict(face)
        # Calculate distances from cluster centers
        dists = np.linalg.norm(self.cluster_center - emb[None, :], axis=1)
        best_cluster = int(np.argmin(dists))
        return best_cluster, dists



def test_open_set_unknowns(training_folders, test_imgs):
    """
    Train the recognizer with all images in all folders in training_folders (a list of folders).
    Evaluate on test_imgs as before.
    """
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    # Train: loop through all folders in training_folders
    for folder in training_folders:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue

            face = detector.detect_face(img)
            if face is None:
                continue

            aligned = detector.align_face(face.image, face.rect)
            recognizer.partial_fit(aligned, label=folder)

    # Evaluate with different metrics
    modes = {
        "distance_only": {"use_distance_limit": True, "use_prob_limit": False},
        "probability_only": {"use_distance_limit": False, "use_prob_limit": True},
        "combined": {"use_distance_limit": True, "use_prob_limit": True},
    }

    for mode, flags in modes.items():
        print(f"\n=== Mode: {mode} ===")
        for fname in sorted(os.listdir(test_imgs)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(test_imgs, fname))
            if img is None:
                continue
            try:
                face = detector.detect_face(img)
            except:
                continue
            if face is None:
                continue
            aligned = detector.align_face(face.image, face.rect)
            pred_label, prob, dist = recognizer.predict(aligned, **flags)
            print(f"{fname:20s} -> Pred: {pred_label:10s} | Prob: {prob:.2f} | Dist: {dist:.2f}")


def test_reidentification(folder_a, folder_b, num_clusters=2):

    clustering = FaceClustering(num_clusters=num_clusters)
    detector = FaceDetector()

    # Training
    for folder in [folder_a, folder_b]:
        image_files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        )

        for fname in image_files[:-5]:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue

            face = detector.track_face(img)
            if face is None:
                continue

            clustering.partial_fit(detector.align_face(face.image, face.rect))

    clustering.fit(plot=True) # the clustering is very sensitive to initialization, it influences the number of k-means-steps strongly

    # Reidentification test
    for folder_name, folder in [("Person A", folder_a), ("Person B", folder_b)]:
        print(f"\nTesting reidentification for {folder_name}:")
        image_files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        )

        for fname in image_files[-5:]:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue

            face = detector.detect_face(img)
            if face is None:
                continue

            cluster_id, dists = clustering.predict(detector.align_face(face.image, face.rect))
            print(f"{fname}: cluster={cluster_id}, dists={np.round(dists, 2)}")


if __name__ == "__main__":
    #test_reidentification(
    #    Config.TEST_DATA.joinpath("Al_Pacino"),
    #    Config.TEST_DATA.joinpath("Alan_Ball"),
    #    num_clusters=2,
    #)
    test_open_set_unknowns([Config.TEST_DATA.joinpath("Al_Pacino"), Config.TEST_DATA.joinpath("Alan_Ball"), Config.TEST_DATA.joinpath("Nancy_Sinatra")], Config.TEST_DATA.joinpath("Marina_Silva"))


