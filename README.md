# Object Detection with Gemini API

This directory contains Python scripts for performing 2D spatial understanding and object detection using Google's Gemini API. The scripts detect and extract objects from images and videos, saving the results as bounding boxes and cropped regions of interest (ROIs).

## Scripts

- **LD1.py**: Object detection demo from Google Colab (original reference implementation)
- **LD2.py**: Command-line tool for detecting objects in static images
- **LD3.py**: Command-line tool for detecting objects in videos or camera feeds

## PDF Report Generation

The `PDF/ERC.ipynb` notebook provides functionality to generate comprehensive mission reports from detected features and landmarks. This Jupyter notebook creates professional PDF reports that include:

- Mission overview and team information
- Environmental context with GPS coordinates and location mapping
- Detailed analysis of detected features using AI-powered descriptions
- Standardized feature documentation with unique IDs
- Technical metadata for each captured element

## Prerequisites

- Python 3.7+
- Google API Key for Gemini (set as environment variable)
- Required packages (install from requirements.txt)

## Setup

1. Get a Google API key from [AI Studio](https://aistudio.google.com/) or [Google AI for Developers](https://ai.google.dev/)
2. Set your API key as an environment variable:

   ```
   # On Windows
   set GOOGLE_API_KEY=your_key_here # for Command Prompt
   $env:GOOGLE_API_KEY="your_key_here" # for PowerShell

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

### Generating PDF Reports (ERC.ipynb)

Open the `PDF/ERC.ipynb` notebook in Jupyter Lab or VS Code and run all cells to generate a comprehensive mission report:

1. Place your feature images in the `PDF/features/` directory
2. Ensure you have a world map image named `map.jpg` in the PDF directory
3. Create a mission description text file named `tet.txt`
4. Run the notebook to generate `ERC.pdf` with:
   - Automated feature analysis using Gemini API
   - GPS location mapping with latitude/longitude markers
   - Professional formatting with unique feature IDs
   - Environmental overview and scientific assessments

**Required files for PDF generation:**

- `PDF/features/`: Directory containing feature images (JPG, PNG, JPEG)
- `PDF/map.jpg`: World map for location visualization
- `PDF/tet.txt`: Mission description text
- `PDF/Mind_cloud_.jpg`: Team logo/image

## How It Works

### Object Detection Scripts

These scripts use Google's Gemini API to perform advanced object detection:

1. Images are sent to the Gemini model (2.5-flash)
2. The model returns JSON data containing bounding boxes for detected objects
3. The scripts process this data to extract the regions of interest
4. For videos, multithreading is used to improve processing efficiency

### PDF Report Generation

The Jupyter notebook automates scientific report creation:

1. Collects feature images from the designated directory
2. Uses geocoding to determine current location coordinates
3. Generates location-aware maps with latitude/longitude markers
4. Employs Gemini AI to analyze each feature image for scientific significance
5. Compiles everything into a professional PDF report with standardized formatting

## Notes and Limitations

### Object Detection

- Processing speed depends on your network connection and Gemini API response time
- For videos, a higher `--skip` value will process faster but analyze fewer frames
- API key environment variable must be set before running the scripts
- The models are optimized to detect man-made objects and exclude natural elements

### PDF Report Generation

- Requires additional packages: `fpdf`, `transformers`, `geocoder`, `python-dotenv`
- Internet connection needed for geocoding and AI analysis
- Generated reports include GPS coordinates based on your current location
- Feature images are automatically resized and standardized for consistent formatting
- Each feature receives a unique randomized ID for documentation purposes
