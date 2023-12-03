import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Function to perform object identification and calculate metrics for a scene
def identify_objects(scene_image_path, object_images_path, threshold=50):
    # Read the scene image
    scene_image = cv2.imread(scene_image_path)
    gray_scene = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

    # Initialize lists to store evaluation metrics
    true_positives, false_positives, true_negatives, false_negatives = [], [], [], []

    for i, object_image_path in enumerate(object_images_path):
        # Read the object image
        object_image = cv2.imread(object_image_path)
        gray_object = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors for both scene and object images
        orb = cv2.ORB_create()
        keypoints_scene, descriptors_scene = orb.detectAndCompute(gray_scene, None)
        keypoints_object, descriptors_object = orb.detectAndCompute(gray_object, None)

        # Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_object, descriptors_scene)

        # Filter matches based on distance
        good_matches = [m for m in matches if m.distance < threshold]

        # Evaluate metrics
        if i + 1 in true_objects_in_scene:
            true_positives.append(len(good_matches))
            false_negatives.append(len(matches) - len(good_matches))
            true_negatives.append(0)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_negatives.append(0)
            true_negatives.append(len(matches) - len(good_matches))
            false_positives.append(len(good_matches))

    # Calculate precision, recall, F1-score, and accuracy
    precision, recall, fscore, _ = precision_recall_fscore_support(
        true_objects_in_scene,
        [1 if tp > 0 else 0 for tp in true_positives],
        average='binary',
    )
    accuracy = accuracy_score(true_objects_in_scene, [1 if tp > 0 else 0 for tp in true_positives])

    return {
        'True Positives': true_positives,
        'False Positives': false_positives,
        'True Negatives': true_negatives,
        'False Negatives': false_negatives,
        'Precision': precision,
        'Recall': recall,
        'F1-score': fscore,
        'Accuracy': accuracy,
    }

# Example usage
scene_image_path = 'Scenes/scene.jpg'  # Replace with the actual path
object_images_path = ['Objects/bag.jpg', 'Objects/obj.jpg']  # Replace with the actual paths
true_objects_in_scene = [1, 0]  # Replace with the actual ground truth information

result = identify_objects(scene_image_path, object_images_path)
print(result)
