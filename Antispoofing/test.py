# test.py
import os
import cv2
import numpy as np
import warnings
import time
import torch

from Antispoofing.src.anti_spoof_predict import AntiSpoofPredict
from Antispoofing.src.generate_patches import CropImage
from Antispoofing.src.utility import parse_model_name

warnings.filterwarnings('ignore')

def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print(f"Image aspect ratio invalid: {width}/{height}. Expected 3/4.")
        return False
    return True


def test(image, model_dir, device_id=None):
   
    # Auto-select device if device_id is None
    if device_id is None:
        if torch.cuda.is_available():
            device_id = 0  # Default to first GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device_id = -1  # Assume -1 indicates CPU; adjust based on AntiSpoofPredict requirements
            print("Using CPU")

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    print("Original image shape:", image.shape)
    image = cv2.resize(image, (int(image.shape[0] * 3 / 4), image.shape[0]))
    print("Resized image shape:", image.shape)
    
    if not check_image(image):
        print("Image check failed.")
        return None
    
    image_bbox = model_test.get_bbox(image)
    print("Bounding box:", image_bbox)
    
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        pred = model_test.predict(img, os.path.join(model_dir, model_name))
        prediction += pred
        test_speed += time.time() - start
        print(f"Model {model_name} prediction: {pred}")

    label = np.argmax(prediction)
    print(f"Final prediction: {prediction}, Label: {label}")
    return label

if __name__ == "__main__":
    import argparse
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="model_lib used to test")
    parser.add_argument("--image_name", type=str, default="image_F1.jpg", help="image used to test")
    args = parser.parse_args()
    image = cv2.imread(args.image_name)
    result = test(image, args.model_dir, args.device_id)
    print("Result:", result)
