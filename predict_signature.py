import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import shutil
import sys
import pathlib
from sklearn.metrics import classification_report
from contextlib import redirect_stdout
import io


# Adjust pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_model(model_path):
    """
    Load the YOLOv5 model with suppressed output.

    Parameters:
    - model_path (str): Path to the YOLO model weights.

    Returns:
    - model: Loaded YOLOv5 model.
    """
    # Suppress output during model loading
    with io.StringIO() as buf, redirect_stdout(buf):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False,verbose=False)
    return model

def make_prediction(model, image_path, confidence_threshold=0.94):
    """
    Perform prediction on an image and check if it meets the confidence threshold.

    Parameters:
    - model: Loaded YOLO model.
    - image_path (str): Path to the image file.
    - confidence_threshold (float): Confidence threshold to consider a detection as "with signature".

    Returns:
    - detected (bool): True if a signature is detected with confidence above the threshold, False otherwise.
    - results: Raw prediction results from the model.
    """
    # Perform prediction on the image
    results = model(image_path)

    # Check detection confidence and apply the threshold
    detected = False
    for *box, conf, cls in results.xyxy[0]:  # results.xyxy contains [x1, y1, x2, y2, confidence, class]
        if conf > confidence_threshold:  # Confidence threshold
            detected = True
            break

    return detected, results

def evaluate_prediction(detected, results, image_path, save_predictions_path):
    """
    Evaluate the prediction by comparing it to the ground truth label.
    Save the image if the prediction is incorrect.

    Parameters:
    - detected (bool): Prediction result from `make_prediction`.
    - results: Raw prediction results from the model.
    - image_path (str): Path to the image file.
    - save_predictions_path (str): Directory to save incorrect predictions.
    """
    # Determine the ground truth label based on the filename
    if '_1' in image_path:
        y_true = 1  # Signature present
    elif '_2' in image_path or '_0' in image_path:
        y_true = 0  # No signature
    else:
        print("The image filename does not have a clear label. Skipping.")
        exit()

    # Predicted label based on detection result
    y_pred = [1 if detected else 0]

    # Save the image, indicating correct or incorrect prediction in the filename
    new_filename = f"{os.path.basename(image_path).split('.')[0]}_detected_{y_pred[0]}_actual_{y_true}.jpg"
    if y_true != y_pred[0]:  # If the prediction is incorrect
        results.save(save_dir=os.path.join(save_predictions_path, new_filename))
        print(f"Incorrect prediction saved as {new_filename}.")
    else:
        results.save(save_dir=os.path.join(save_predictions_path, new_filename))
        print(f"Correct prediction saved as {new_filename}.")

if __name__ == '__main__':
    # Example usage
    model_path = 'runs/train/exp24/weights/best.pt'  # Path to the trained YOLO model weights
    image_path = '23_0.jpeg'  # Path to the single image
    save_predictions_path = './prediction_result'  # Directory to save predictions

    # Load the model with suppressed output
    model = load_model(model_path)

    # Make a prediction and evaluate it
    detected, results = make_prediction(model, image_path)
    evaluate_prediction(detected, results, image_path, save_predictions_path)
