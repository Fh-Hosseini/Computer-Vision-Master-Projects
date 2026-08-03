from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(
        self, tm_window_size: int = 100, tm_threshold: float = 0.5, aligned_image_size: int = 224
    ) -> None:
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference: Optional[FaceDetectionResult] = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # How much we move the bounding box to detect faces using template matching
        self.tm_window_size = tm_window_size

        self.tm_threshold = tm_threshold


    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:

        # Check if any reference exists from before. If not, detect the face using detect_face() function.
        if self.reference is None:
            detection = self.detect_face(image)
            if detection is None:
                return None
            self.reference = detection
            return detection

        # If a reference exists, try to get a template of face from the reference by cropping.
        template = self.crop_face(self.reference.image, self.reference.rect)
        temp_height, temp_width = template.shape[:2]

        # Reference bounding box
        ref_x, ref_y, ref_width, ref_height = self.reference.rect

        # Define search region around the reference image based on tm_window_size as margin
        x_start = max(ref_x - self.tm_window_size, 0)
        y_start = max(ref_y - self.tm_window_size, 0)
        x_end = min(ref_x + ref_width + self.tm_window_size, image.shape[1])
        y_end = min(ref_y + ref_height + self.tm_window_size, image.shape[0])

        # Before getting the searching area, check if it has a proper size based on the template size
        if y_end - y_start < temp_height or x_end - x_start < temp_width:
            detection = self.detect_face(image)
            if detection is not None:
                self.reference = detection
            return detection

        # Define a searching area from the frame into a new image
        search_region = image[y_start:y_end, x_start:x_end]

        # Apply template matching method to find the face template in new frame
        result = cv2.matchTemplate(
            search_region,
            template,
            cv2.TM_CCOEFF_NORMED,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if the similarity is high enough using the similarity threshold
        # if there is not enough similarity, try to detect the face again.
        if max_val < self.tm_threshold:
            detection = self.detect_face(image)
            if detection is None:
                return None
            self.reference = detection
            return detection


        # Compute the new positions for bounding box of reference image
        top_left = max_loc
        new_x = x_start + top_left[0]
        new_y = y_start + top_left[1]
        new_rect = (new_x, new_y, ref_width, ref_height) # issue: template matching does not adjust the size of the bounding box

        # Align face at new position using predefined function
        aligned = self.align_face(image, new_rect)

        # Update reference based on new positions
        self.reference = FaceDetectionResult(
            image=image,
            rect=new_rect,
            aligned=aligned,
        )

        return self.reference

    # Face detection in a new image.
    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        # Retrieve all detectable faces in the given image.
        try:
            if not (
                    detections := self.detector.detect_faces(image, threshold_pnet=0.85, threshold_rnet=0.9)
            ):
                self.reference = None
                return None
        except:
            return None

        # Select face with the largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return FaceDetectionResult(rect=face_rect, image=image, aligned=aligned)

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]