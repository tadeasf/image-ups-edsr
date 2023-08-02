import os
import time
import json
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from realesrgan_ncnn_py import Realesrgan
import cv2

# Define the model configurations with their corresponding indices
models = {
    0: {
        "param": "realesr-animevideov3-x2.param",
        "bin": "realesr-animevideov3-x2.bin",
        "scale": 2,
        "max_long_side": 7000,
    },
    1: {
        "param": "realesr-animevideov3-x3.param",
        "bin": "realesr-animevideov3-x3.bin",
        "scale": 3,
        "max_long_side": 7000,
    },
    2: {
        "param": "realesr-animevideov3-x4.param",
        "bin": "realesr-animevideov3-x4.bin",
        "scale": 4,
        "max_long_side": 7000,
    },
    3: {
        "param": "realesrgan-x4plus-anime.param",
        "bin": "realesrgan-x4plus-anime.bin",
        "scale": 4,
        "max_long_side": 7000,
    },
    4: {
        "param": "realesrgan-x4plus.param",
        "bin": "realesrgan-x4plus.bin",
        "scale": 4,
        "max_long_side": 7000,
    },
}

# Ask user for the maximum long side size
max_long_side_input = int(
    input("Enter the maximum long side size for the upscaled images (e.g., 7000): ")
)

# Update the models with the user-specified max long side size
for model_info in models.values():
    model_info["max_long_side"] = max_long_side_input

# Ask user for input directory
input_dir = input("Enter the input directory path: ")
# Ask user for output directory
output_dir = input("Enter the output directory path: ")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ask user to choose the model and print the available options
print("Available models:")
for idx, model_info in models.items():
    print(f"{idx}: Scale {model_info['scale']} - Model: {model_info['param']}")

model_idx = int(input("Enter the model index you want to use: "))
model_config = models[model_idx]

# Initialize the Realesrgan model
realesrgan = Realesrgan(
    gpuid=0,
    model=model_config["scale"],
)


def downscale_image_with_sharpening(image, max_long_side, upscale_factor):
    effective_max_long_side = (
        max_long_side / upscale_factor
    )  # Divide instead of multiply
    width, height = image.size
    if max(width, height) * upscale_factor > effective_max_long_side:
        scale_factor = effective_max_long_side / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # Use OpenCV for resizing with Lanczos interpolation
        image_array = np.array(image)
        image_array = cv2.resize(
            image_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )
        image = Image.fromarray(image_array)
        enhancer = ImageEnhance.Sharpness(image)
        sharpened_image = enhancer.enhance(2.0)
        return sharpened_image
    return image  # Return the original image if no downscaling is necessary


# Function to upscale the image and measure time taken
def upscale_image_and_measure_time(image, output_path):
    start_time = time.time()
    image = np.array(image)
    image = realesrgan.process_cv2(image)
    # Convert back to RGB from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imencode(".jpg", image)[1].tofile(output_path)
    end_time = time.time()
    return end_time - start_time


# Dictionary to store image filenames and corresponding upscale times
upscale_times = {}

# Recursively find every image in the input directory with the specified extensions
valid_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
total_upscale_time = 0

for root, _, files in os.walk(input_dir):
    for file in files:
        if any(file.endswith(ext) for ext in valid_extensions):
            input_image_path = os.path.join(root, file)
            output_image_path = os.path.join(output_dir, file)

            # Open the image and downscale if necessary based on the selected model
            image = Image.open(input_image_path).convert("RGB")
            max_long_side = model_config["max_long_side"]
            image = downscale_image_with_sharpening(
                image, max_long_side, model_config["scale"]
            )

            # Upscale the downscaled and sharpened image using Realesrgan and measure time taken
            time_taken = upscale_image_and_measure_time(image, output_image_path)
            upscale_times[file] = time_taken
            total_upscale_time += time_taken

# Sort image filenames based on the corresponding upscale times (longest to shortest)
sorted_upscale_times = sorted(upscale_times.items(), key=lambda x: x[1], reverse=True)

# Generate JSON output with required information
output_json = {
    "total_upscale_time": str(
        time.strftime("%H:%M:%S", time.gmtime(total_upscale_time))
    ),
    "num_images_upscaled": len(sorted_upscale_times),
    "image_upscale_times": {
        filename: str(time.strftime("%H:%M:%S", time.gmtime(time_taken)))
        for filename, time_taken in sorted_upscale_times
    },
}

# Ask user for the desired JSON file name
json_filename = input("Enter the desired name for the JSON file (without extension): ")

# Append ".json" extension to the filename
json_filename += ".json"

# Save the JSON output to the specified file in the root + "upscale-data" directory
output_json_path = os.path.join("upscale-data", json_filename)
if not os.path.exists("upscale-data"):
    os.makedirs("upscale-data")
with open(output_json_path, "w") as f:
    json.dump(output_json, f, indent=4)

print(f"Upscaling process completed. JSON output saved to {json_filename}.")
