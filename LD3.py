# -*- coding: utf-8 -*-
"""
2D spatial understanding in videos with Gemini.

This script takes a video path or camera index as a command-line argument,
detects objects in the video frames using the Gemini API, draws bounding boxes
around them, and saves the output video to a 'landmarks' directory.

Usage:
    python LD3.py "path/to/your/video.mp4"
    or
    python LD3.py 0 (for the default camera)

Make sure to set the GOOGLE_API_KEY environment variable before running.
"""

import sys
import os
import json
import time
from PIL import Image, ImageDraw, ImageFont, ImageColor
import google.generativeai as genai
from google.generativeai import types
import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

def parse_json_output(json_output: str):
    """
    Parses the JSON output from the model, removing markdown fencing.
    """
    if not json_output:
        return None
    if "```json" in json_output:
        json_output = json_output.split("```json")[1].split("```")[0]
    return json_output

def draw_bounding_boxes_on_frame(frame, bounding_boxes_str):
    """
    Draws bounding boxes on an OpenCV frame.
    """
    if not bounding_boxes_str:
        return frame

    height, width, _ = frame.shape
    
    colors = [color for color, code in ImageColor.colormap.items() if color not in ['black', 'white']]
    if not colors:
        colors = ['red', 'green', 'blue', 'yellow', 'purple']

    try:
        bounding_boxes_json = parse_json_output(bounding_boxes_str)
        if not bounding_boxes_json:
            return frame
        bounding_boxes = json.loads(bounding_boxes_json)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing bounding box JSON: {e}")
        print(f"Received: {bounding_boxes_str}")
        return frame

    for i, bounding_box in enumerate(bounding_boxes):
        color_name = colors[i % len(colors)]
        color_rgb = ImageColor.getrgb(color_name)
        # Convert RGB to BGR for OpenCV
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

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

        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color_bgr, 4)

    return frame

def process_frame_in_thread(frame, model, prompt, frame_count, source_name, output_dir, lock, shared_data, video_time_str):
    """
    Processes a single frame in a separate thread to detect objects and save ROIs.
    """
    print(f"Detecting objects in frame #{frame_count} (Video time: {video_time_str})...")

    # Convert frame for Gemini
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

    try:
        response = model.generate_content(
            contents=[prompt, pil_img],
            generation_config=types.GenerationConfig(
                temperature=0.5,
            ),
            request_options={'timeout': 60.0}
        )
        
        with lock:
            shared_data['last_bounding_boxes'] = response.text
        
        print(f"Object detection complete for frame #{frame_count}.")

        # Save the ROIs as images
        if response.text:
            try:
                bounding_boxes_json = parse_json_output(response.text)
                if bounding_boxes_json:
                    bounding_boxes = json.loads(bounding_boxes_json)
                    height, width, _ = frame.shape
                    for i, bounding_box in enumerate(bounding_boxes):
                        box = bounding_box.get("box_2d")
                        if not box or len(box) != 4:
                            continue

                        # Calculate absolute coordinates
                        abs_y1 = int(box[0] / 1000 * height)
                        abs_x1 = int(box[1] / 1000 * width)
                        abs_y2 = int(box[2] / 1000 * height)
                        abs_x2 = int(box[3] / 1000 * width)

                        if abs_x1 > abs_x2:
                            abs_x1, abs_x2 = abs_x2, abs_x1
                        if abs_y1 > abs_y2:
                            abs_y1, abs_y2 = abs_y2, abs_y1
                        
                        # Crop the ROI from the original frame
                        roi = frame[abs_y1:abs_y2, abs_x1:abs_x2]

                        # Save the ROI image
                        if roi.size > 0:
                            roi_filename = f"{source_name}_frame_{frame_count}_roi_{i}.png"
                            roi_path = os.path.join(output_dir, roi_filename)
                            cv2.imwrite(roi_path, roi)
                            print(f"ROI for frame {frame_count} saved to {roi_path}")

            except Exception as e:
                print(f"Could not save ROI for frame {frame_count}: {e}")

    except Exception as e:
        print(f"Error during API call for frame #{frame_count}: {e}")


def main(video_source, skip_frames):
    """
    Main function to process the video.
    """
    # --- Configuration ---
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    model_name = "gemini-2.5-flash"
    prompt = """
    Detect the 2d bounding boxes of the man-made objects and nearby objects only.
    Ignore objects that are far away or not clearly visible.
    Ignore rocks, cairn, sand or any natural elements and far objects and any part that can't exist alone.
    If the object is too close to the camera (not clear and not focused), ignore it.
    Ignore objects that are part of a larger object. (e.g., a tire is part of a car, ignore the tire)
    Return the bounding boxes as a JSON array.
    """

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

    # --- Video Input Setup ---
    try:
        video_source_int = int(video_source)
        is_camera = True
        cap = cv2.VideoCapture(video_source_int)
        source_name = f"camera_{video_source_int}"
    except ValueError:
        is_camera = False
        cap = cv2.VideoCapture(video_source)
        source_name = os.path.basename(video_source).split('.')[0]

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        sys.exit(1)

    # --- Output Video Setup ---
    output_dir = "landmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # --- Processing Loop ---
    shared_data = {'last_bounding_boxes': None}
    lock = threading.Lock()
    frame_count = 0

    print("Processing video... This may take a while. Press Ctrl+C to stop.")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get current video time
            video_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            video_time_str = time.strftime('%H:%M:%S', time.gmtime(video_timestamp_ms / 1000))

            # Only call API on the first frame and after skipping 'skip_frames'
            if (frame_count - 1) % (skip_frames + 1) == 0:
                # Submit the frame processing to the thread pool
                executor.submit(process_frame_in_thread, frame.copy(), model, prompt, frame_count, source_name, output_dir, lock, shared_data, video_time_str)

            # Draw boxes on the current frame using most recent detection
            with lock:
                current_boxes = shared_data['last_bounding_boxes']
            
            processed_frame = draw_bounding_boxes_on_frame(frame.copy(), current_boxes)
            
            # Display the processed frame (Removed due to headless environment incompatibility)
            # cv2.imshow('Video Object Detection', processed_frame)
            
            # Video saving is disabled
            # output_writer.write(processed_frame)
            
            # Check for exit (Removed as it depends on cv2.imshow)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    end_time = time.time()
    total_processing_time = end_time - start_time

    # --- Cleanup ---
    cap.release()
    # output_writer.release()  # Video writer is disabled
    # cv2.destroyAllWindows() # Not needed as no windows are created
    print("Processing complete!")
    print(f"Total processing time: {total_processing_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D spatial understanding in videos with Gemini.")
    parser.add_argument("video_source", help="Path to the video file or camera index (e.g., 0).")
    parser.add_argument("--skip", type=int, default=29, help="Number of frames to skip between detections. Default is 29 (detect roughly once per second for a 30fps video).")
    
    args = parser.parse_args()
    
    main(args.video_source, args.skip)

