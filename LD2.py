# -*- coding: utf-8 -*-
"""
2D spatial understanding with Gemini.

This script takes an image path as a command-line argument,
detects objects in the image using the Gemini API, draws bounding boxes
around them, and saves the output image to a 'landmarks' directory.

Usage:
    python LD2.py "path/to/your/image.jpg"

Make sure to set the GOOGLE_API_KEY environment variable before running.
"""

import sys
import os
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import google.generativeai as genai
from google.generativeai import types

def parse_json_output(json_output: str):
    """
    Parses the JSON output from the model, removing markdown fencing.
    """
    if "```json" in json_output:
        json_output = json_output.split("```json")[1].split("```")[0]
    return json_output

def plot_bounding_boxes(im, bounding_boxes_str, output_path):
    """
    Plots bounding boxes on an image and saves it.
    """
    width, height = im.size
    draw = ImageDraw.Draw(im)

    colors = [colorname for colorname, colorcode in ImageColor.colormap.items()]
    if not colors:
        colors = ['red', 'green', 'blue', 'yellow', 'purple']

    try:
        bounding_boxes_json = parse_json_output(bounding_boxes_str)
        bounding_boxes = json.loads(bounding_boxes_json)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing bounding box JSON: {e}")
        print(f"Received: {bounding_boxes_str}")
        return

    try:
        # Try to use a system font, fallback to default
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    for i, bounding_box in enumerate(bounding_boxes):
        color = colors[i % len(colors)]

        box = bounding_box.get("box_2d")
        if not box or len(box) != 4:
            continue

        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

    im.save(output_path)
    print(f"Output image saved to: {output_path}")

def main(image_path):
    """
    Main function to process the image.
    """
    # --- Configuration ---
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    model_name = "gemini-2.5-flash-preview-05-20"
    prompt = "Detect the 2d bounding boxes of the man-made and close objects only (ignore objects that are part of a larger object, rocks, sand or any natural elements)."

    bounding_box_system_instructions = """
    Return bounding boxes as a JSON array. Never return masks or code fencing. Limit to 25 objects.
      """

    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
    ]

    # --- Initialization ---
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name, 
                                     system_instruction=bounding_box_system_instructions,
                                     safety_settings=safety_settings)
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        sys.exit(1)

    # --- Image Loading ---
    try:
        im = Image.open(image_path)
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    # --- API Call ---
    print("Detecting objects in the image...")
    try:
        response = model.generate_content(
            contents=[prompt, im],
            generation_config=types.GenerationConfig(
                temperature=0.5,
            ),
            safety_settings=safety_settings,
            request_options={'timeout': 600.0}
        )
        print("Object detection complete.")
        print("Response from model:\n", response.text)
    except Exception as e:
        print(f"Error during API call: {e}")
        sys.exit(1)

    # --- Output Handling ---
    output_dir = "landmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_landmarks{ext}"
    output_path = os.path.join(output_dir, output_filename)

    plot_bounding_boxes(im.copy(), response.text, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python LD2.py \"<image_path>\"")
        sys.exit(1)
    
    image_path_arg = sys.argv[1]
    main(image_path_arg)
