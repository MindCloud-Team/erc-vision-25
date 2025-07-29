# -*- coding: utf-8 -*-
"""
Real-time 2D spatial understanding with Gemini.

This script captures live camera feed, detects objects in frames using Gemini API,
draws bounding boxes around them, and allows user interaction with keyboard:
- 'k': Save screenshot to saved_frames directory (with bounding boxes)
- 's': Skip processing current frame
- 'q': Quit
"""

import os
import json
import time
import cv2
import numpy as np
import argparse
from PIL import Image, ImageColor
import google.generativeai as genai
from google.generativeai import types
import threading
import queue
import sys
import signal

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C signal."""
    print("\nReceived interrupt signal. Shutting down...")
    shutdown_event.set()

def parse_json_output(json_output: str):
    """Parses JSON output from model, removing markdown fencing."""
    if not json_output:
        return None
    if "```json" in json_output:
        json_output = json_output.split("```json")[1].split("```")[0]
    return json_output

def draw_bounding_boxes_on_frame(frame, bounding_boxes_str):
    """Draws bounding boxes on OpenCV frame with labels."""
    if not bounding_boxes_str:
        return frame

    height, width, _ = frame.shape
    
    colors = [color for color, code in ImageColor.colormap.items() 
             if color not in ['black', 'white'] and not color.startswith('grey')]
    if not colors:
        colors = ['red', 'green', 'blue', 'yellow', 'purple']

    try:
        bounding_boxes_json = parse_json_output(bounding_boxes_str)
        if not bounding_boxes_json:
            return frame
        bounding_boxes = json.loads(bounding_boxes_json)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSON parse error: {e}")
        return frame

    for i, bounding_box in enumerate(bounding_boxes):
        if i >= len(colors):
            break
            
        color_name = colors[i % len(colors)]
        try:
            color_rgb = ImageColor.getrgb(color_name)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        except ValueError:
            color_bgr = (0, 255, 0)  # Fallback to green

        box = bounding_box.get("box_2d")
        if not box or len(box) != 4:
            continue

        # Convert relative coordinates to absolute pixels
        abs_y1 = int(box[0] / 1000 * height)
        abs_x1 = int(box[1] / 1000 * width)
        abs_y2 = int(box[2] / 1000 * height)
        abs_x2 = int(box[3] / 1000 * width)

        # Ensure valid coordinates
        abs_x1, abs_x2 = sorted([abs_x1, abs_x2])
        abs_y1, abs_y2 = sorted([abs_y1, abs_y2])
        
        # Skip invalid boxes
        if abs_x1 >= abs_x2 or abs_y1 >= abs_y2:
            continue
        
        # Draw rectangle
        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color_bgr, 2)
        
        # Draw label
        label = bounding_box.get("object", "object")
        cv2.putText(frame, label, (abs_x1, abs_y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    return frame

def process_frame(model, prompt, frame, frame_count, results_queue):
    """Processes frame to detect objects using Gemini API."""
    # Check if shutdown was requested
    if shutdown_event.is_set():
        return
        
    # Convert frame for Gemini
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

    try:
        response = model.generate_content(
            contents=[prompt, pil_img],
            generation_config=types.GenerationConfig(temperature=0.5),
            request_options={'timeout': 5.0}  # Shorter timeout for real-time
        )
        
        if response.text and not shutdown_event.is_set():
            results_queue.put((frame_count, response.text))
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"API error: {e}")

def main(camera_index=0, skip_frames=20):
    """Main function to process live camera feed."""
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # --- Configuration ---
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    model_name = "gemini-1.5-flash"
    prompt = """
    Detect the 2d bounding boxes of the man-made objects and nearby objects only.
    Return bounding boxes as a JSON array with 'object' label and 'box_2d' in [x1, y1, x2, y2] format.
    Use relative coordinates (0-1000) for the box positions.
    """

    # --- Initialization ---
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name, 
            system_instruction="Return bounding boxes as JSON array. Limit to 10 objects."
        )
    except Exception as e:
        print(f"GenAI init error: {e}")
        return

    # --- Camera Setup ---
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index: {camera_index}")
        return

    # Create output directory
    output_dir = "saved_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Screenshots will be saved to: {os.path.abspath(output_dir)}")
    
    # Create a queue for processing results
    results_queue = queue.Queue()
    last_boxes_str = None
    last_boxes_frame = None
    frame_count = 0
    processing_thread = None
    
    # Try to create GUI window
    try:
        cv2.namedWindow("Live Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Object Detection", 1000, 700)
        gui_supported = True
    except cv2.error:
        print("GUI not supported - running in headless mode")
        gui_supported = False
    
    print("Starting live camera feed...")
    print("Controls: 'k'=Save screenshot, 's'=Skip frame, 'q'=Quit, 'Ctrl+C'=Force quit")
    
    # Main loop
    try:
        while not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
                
            display_frame = frame.copy()
            frame_count += 1
            
            # Process frame conditionally
            if frame_count % (skip_frames + 1) == 0:
                if processing_thread is None or not processing_thread.is_alive():
                    processing_thread = threading.Thread(
                        target=process_frame,
                        args=(model, prompt, frame.copy(), frame_count, results_queue)
                    )
                    processing_thread.daemon = True  # Make thread daemon so it exits with main
                    processing_thread.start()
            
            # Check for new results
            while not results_queue.empty():
                try:
                    result_frame_count, boxes_str = results_queue.get_nowait()
                    last_boxes_str = boxes_str
                    last_boxes_frame = frame.copy()  # Save frame for drawing boxes
                    print(f"New detection results for frame #{result_frame_count}")
                except queue.Empty:
                    break
            
            # Draw bounding boxes if available
            if last_boxes_str:
                # Draw boxes on the current display frame
                display_frame = draw_bounding_boxes_on_frame(display_frame.copy(), last_boxes_str)
            
            # Show frame in GUI if supported
            if gui_supported:
                # Add info overlay
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "k:Save  s:Skip  q:Quit", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show processing status
                status = "Processing..." if processing_thread and processing_thread.is_alive() else "Ready"
                cv2.putText(display_frame, f"Status: {status}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Display frame
                cv2.imshow("Live Object Detection", display_frame)
            
            # Handle key presses with shorter wait time for better responsiveness
            if gui_supported:
                key = cv2.waitKey(1) & 0xFF
            else:
                # Shorter wait in headless mode for better Ctrl+C response
                key = cv2.waitKey(1) & 0xFF
                
            if key == ord('q'):  # Quit
                break
            elif key == ord('k'):  # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}_{frame_count}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Always save with bounding boxes if available
                if last_boxes_str and last_boxes_frame is not None:
                    # Create a clean copy of the frame with boxes
                    save_frame = last_boxes_frame.copy()
                    save_frame = draw_bounding_boxes_on_frame(save_frame, last_boxes_str)
                    cv2.imwrite(filepath, save_frame)
                    print(f"Saved screenshot with boxes: {filepath}")
                else:
                    # Save current frame if no boxes available
                    cv2.imwrite(filepath, frame)
                    print(f"Saved screenshot without boxes: {filepath}")
            elif key == ord('s'):  # Skip processing
                print("Skipping frame processing")
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    finally:
        # Cleanup
        shutdown_event.set()
        cap.release()
        if gui_supported:
            cv2.destroyAllWindows()
        if processing_thread and processing_thread.is_alive():
            print("Waiting for processing thread to finish...")
            processing_thread.join(timeout=2.0)  # Wait max 2 seconds
        print("Application closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time object detection with Gemini.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--skip", type=int, default=20, 
                       help="Frames to skip between detections (default: 20)")
    
    args = parser.parse_args()
    
    try:
        main(camera_index=args.camera, skip_frames=args.skip)
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        sys.exit(0)