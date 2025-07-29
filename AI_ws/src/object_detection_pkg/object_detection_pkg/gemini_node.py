# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import json
import os
import sys
import google.generativeai as genai
from google.generativeai import types

# Import our custom messages
from rovers_interfaces.msg import Detection, DetectionList

class GeminiDetectorNode(Node):
    def __init__(self):
        super().__init__('gemini_detector_node')

        # Declare parameter for the API Key
        self.declare_parameter('google_api_key', '')
        self.api_key = self.get_parameter('google_api_key').get_parameter_value().string_value

        if not self.api_key:
            self.get_logger().error("Google API Key parameter 'google_api_key' is not set! Shutting down.")
            sys.exit(1)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Create Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/panther/camera_front/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Subscribed to /panther/camera_front/image_raw")

        # Create Publisher for our custom message
        self.publisher_ = self.create_publisher(DetectionList, 'gemini_detections', 10)
        self.get_logger().info("Publishing detections to /gemini_detections")

        # --- Gemini Model Configuration ---
        self.model = self.setup_gemini_model()

    def setup_gemini_model(self):
        """Initializes and configures the Gemini model."""
        try:
            genai.configure(api_key=self.api_key)
            model_name = "gemini-1.5-flash-preview-0514" # Using 1.5 as it's often better/faster
            bounding_box_system_instructions = """
            Return bounding boxes as a JSON array. Never return masks or code fencing. Limit to 25 objects.
            Each object should have a "label" and a "box_2d" array with four numbers [y_min, x_min, y_max, x_max]
            normalized to 1000.
            """
            safety_settings = [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            model = genai.GenerativeModel(
                model_name,
                system_instruction=bounding_box_system_instructions,
                safety_settings=safety_settings
            )
            self.get_logger().info(f"Gemini model '{model_name}' initialized successfully.")
            return model
        except Exception as e:
            self.get_logger().error(f"Error initializing GenAI client: {e}")
            sys.exit(1)

    def parse_json_output(self, json_output: str):
        """Parses the JSON output from the model, removing markdown fencing."""
        if "```json" in json_output:
            json_output = json_output.split("```json")[1].split("```")[0]
        return json.loads(json_output)

    def image_callback(self, msg: Image):
        """Callback function for the image subscriber."""
        self.get_logger().info('Received image frame. Processing with Gemini...')
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert OpenCV image (BGR) to PIL image (RGB)
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            pil_image.thumbnail([1024, 1024], PILImage.Resampling.LANCZOS)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # --- API Call ---
        prompt = "Detect the 2d bounding boxes of the man-made and close objects only (ignore objects that are part of a larger object, rocks, sand or any natural elements)."
        try:
            response = self.model.generate_content(
                contents=[prompt, pil_image],
                generation_config=types.GenerationConfig(temperature=0.2),
                request_options={'timeout': 60.0}
            )
        except Exception as e:
            self.get_logger().error(f"Error during Gemini API call: {e}")
            return
            
        self.get_logger().info('Gemini response received.')
        # self.get_logger().info(f"DEBUG: Gemini Raw Text: {response.text}")

        # --- Process Response and Publish ---
        try:
            bounding_boxes_json = self.parse_json_output(response.text)
            
            detection_list_msg = DetectionList()
            detection_list_msg.header = msg.header # CRITICAL: Use header from the input image

            img_height, img_width = cv_image.shape[:2]

            for bbox_data in bounding_boxes_json:
                box = bbox_data.get("box_2d")
                if not box or len(box) != 4:
                    continue
                
                # Gemini format is [y_min, x_min, y_max, x_max] normalized to 1000
                y1_norm, x1_norm, y2_norm, x2_norm = box

                # Convert normalized (0-1000) coordinates to absolute pixel coordinates
                # and ensure they are within image bounds
                abs_x1 = min(max(int(x1_norm / 1000 * img_width), 0), img_width)
                abs_y1 = min(max(int(y1_norm / 1000 * img_height), 0), img_height)
                abs_x2 = min(max(int(x2_norm / 1000 * img_width), 0), img_width)
                abs_y2 = min(max(int(y2_norm / 1000 * img_height), 0), img_height)

                # Create a single detection message
                detection_msg = Detection()
                detection_msg.label = bbox_data.get("label", "object")
                # Store as [xmin, ymin, xmax, ymax]
                detection_msg.bbox = [abs_x1, abs_y1, abs_x2, abs_y2]
                
                detection_list_msg.detections.append(detection_msg)
            
            # Publish the list of detections
            self.publisher_.publish(detection_list_msg)
            self.get_logger().info(f"Published {len(detection_list_msg.detections)} detections.")

        except (json.JSONDecodeError, IndexError) as e:
            self.get_logger().error(f"Error parsing bounding box JSON: {e}")
            self.get_logger().error(f"Received from model: {response.text}")


def main(args=None):
    rclpy.init(args=args)
    node = GeminiDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
