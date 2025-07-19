# Object Detection with Gemini API

This directory contains Python scripts for performing 2D spatial understanding and object detection using Google's Gemini API. The scripts detect and extract objects from images and videos, saving the results as bounding boxes and cropped regions of interest (ROIs).

## Scripts

- **LD1.py**: Object detection demo from Google Colab (original reference implementation)
- **LD2.py**: Command-line tool for detecting objects in static images
- **LD3.py**: Command-line tool for detecting objects in videos or camera feeds

## Prerequisites

- Python 3.7+
- Google API Key for Gemini (set as environment variable)
- Required packages (install from requirements.txt)

## Setup

1. Get a Google API key from [AI Studio](https://aistudio.google.com/) or [Google AI for Developers](https://ai.google.dev/)
2. Set your API key as an environment variable:

   ```
   # On Windows
   set GOOGLE_API_KEY=your_key_here

   # On macOS/Linux
   export GOOGLE_API_KEY=your_key_here
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Processing Images (LD2.py)

```bash
python LD2.py "path/to/your/image.jpg"
```

This will:

1. Load the specified image
2. Detect objects using Gemini API
3. Draw bounding boxes around detected objects
4. Save the result as "[original_filename]\_landmarks.jpg" in the "landmarks" directory

### Processing Videos (LD3.py)

```bash
# For video files:
python LD3.py "path/to/your/video.mp4" --skip 10

# For camera:
python LD3.py 0 --skip 5
```

Options:

- `--skip N`: Process every N+1 frames (default: 29)

This will:

1. Process the video or camera feed
2. Detect objects in frames at the specified interval
3. Save individual object ROIs as PNG files in the "landmarks" directory
4. Print processing time upon completion


## How It Works

These scripts use Google's Gemini API to perform advanced object detection:

1. Images are sent to the Gemini model (2.5-flash)
2. The model returns JSON data containing bounding boxes for detected objects
3. The scripts process this data to extract the regions of interest
4. For videos, multithreading is used to improve processing efficiency

## Notes and Limitations

- Processing speed depends on your network connection and Gemini API response time
- For videos, a higher `--skip` value will process faster but analyze fewer frames
- API key environment variable must be set before running the scripts
- The models are optimized to detect man-made objects and exclude natural elements
