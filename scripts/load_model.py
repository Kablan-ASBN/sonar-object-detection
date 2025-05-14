# load_model.py
# This script loads the pretrained Faster R-CNN model using Detectron2.
# It returns a predictor object for inference. If run directly, it will also test the model on a real sonar image
# from the training set and visualize the predictions for validation purposes.

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import os


# Step 1: Load the pretrained Faster R-CNN model and return the predictor and config
def load_model(weight_path):
    # Initialize configuration object and load the base model architecture from Detectron2 model zoo
    cfg = get_cfg()
    # Use the COCO Faster R-CNN architecture as base
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Load the model checkpoint
    cfg.MODEL.WEIGHTS = weight_path

    # Set detection threshold to ignore low-confidence predictions
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    # Set the compute device to CPU (change to "cuda" if using a GPU)
    cfg.MODEL.DEVICE = "cpu"

    # Update NUM_CLASSES based on your dataset (default 1 class: "object")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Update if I have more classes

    # Create the predictor object that wraps preprocessing and forward pass logic
    predictor = DefaultPredictor(cfg)

    return predictor, cfg

# Step 2: If this file is run directly, perform inference on a sonar training image
if __name__ == "__main__":
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    # Get the directory of this script to construct relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path to the Seabed.ai pretrained model checkpoint
    MODEL_PATH = os.path.join(SCRIPT_DIR, "../checkpoints/my_frcnn_model.pth")

    # Replace this with the name of any sonar image from the training set (JPEGImages folder)
    TEST_IMAGE_NAME = "Line 2_CNav_500_0_1000_500.jpg"
    TEST_IMAGE_PATH = os.path.join(SCRIPT_DIR, "../data/line2yolo/images", TEST_IMAGE_NAME)

    # Load model and configuration
    predictor, cfg = load_model(MODEL_PATH)

    # Load the test sonar image
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {TEST_IMAGE_PATH}. Please check the filename and path.")

    # Run inference
    outputs = predictor(image)

    # Manual metadata (because dataset not registered)
    MetadataCatalog.get("sonar_dataset").set(thing_classes=["object"])

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("sonar_dataset"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
