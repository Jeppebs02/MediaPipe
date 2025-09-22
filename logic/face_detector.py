from typing import Tuple, Union
import math
import cv2
import numpy as np


class DetectionVisualizer:
    def __init__(self,
                 margin: int = 10,
                 row_size: int = 10,
                 font_size: int = 1,
                 font_thickness: int = 1,
                 text_color: Tuple[int, int, int] = (255, 0, 0)):  # red
        self.margin = margin
        self.row_size = row_size
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.text_color = text_color

    def _normalized_to_pixel_coordinates(
        self,
        normalized_x: float,
        normalized_y: float,
        image_width: int,
        image_height: int
    ) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (
                value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            return None

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def visualize(self, image, detection_result) -> np.ndarray:
        """
        Draws bounding boxes and keypoints on the input image and returns it.

        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to visualize.

        Returns:
            Annotated image with bounding boxes, keypoints, and labels.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width,
                         bbox.origin_y + bbox.height)
            cv2.rectangle(annotated_image, start_point, end_point,
                          self.text_color, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(
                    keypoint.x, keypoint.y, width, height)
                if keypoint_px:  # Only draw if valid
                    cv2.circle(annotated_image, keypoint_px,
                               2, (0, 255, 0), 2)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name or ''
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            text_location = (self.margin + bbox.origin_x,
                             self.margin + self.row_size + bbox.origin_y)

            cv2.putText(annotated_image, result_text, text_location,
                        cv2.FONT_HERSHEY_PLAIN, self.font_size,
                        self.text_color, self.font_thickness)

        return annotated_image
