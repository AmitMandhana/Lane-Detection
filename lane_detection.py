import cv2
import numpy as np
import os
from pathlib import Path

def preprocess_frame(frame):
    """
    Convert frame to grayscale, apply Gaussian blur, and Canny edge detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    """
    Create a mask for the region of interest (ROI) in the frame.
    Focuses on the lower trapezoidal region where lanes are most likely to be.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Define a trapezoidal ROI (adjust these points based on your camera angle)
    bottom_left = (width * 0.1, height)
    bottom_right = (width * 0.9, height)
    top_left = (width * 0.4, height * 0.6)
    top_right = (width * 0.6, height * 0.6)
    
    vertices = np.array([[
        bottom_left,
        top_left,
        top_right,
        bottom_right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    roi = cv2.bitwise_and(edges, mask)
    return roi

def detect_lane_lines(roi):
    """
    Detect lines in the ROI using Hough Transform.
    Returns left and right lane lines.
    """
    height, width = roi.shape
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )
    
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope
            if x2 == x1:
                continue  # Skip vertical lines
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter lines based on slope
            if abs(slope) < 0.5:  # Horizontal lines
                continue
                
            if slope < 0:  # Left lane (negative slope)
                left_lines.append(line[0])
            else:  # Right lane (positive slope)
                right_lines.append(line[0])
    
    return left_lines, right_lines

def average_slope_intercept(lines, frame_shape):
    """
    Average all the lines in the frame to get a single line for left and right lane.
    """
    if lines is None or len(lines) == 0:
        return None
        
    x, y = [], []
    for line in lines:
        x.extend([line[0], line[2]])
        y.extend([line[1], line[3]])
        
    if len(x) == 0:
        return None
        
    # Fit a line through all the points
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate y values for top and bottom of ROI
    y1 = frame_shape[0]  # Bottom of the frame
    y2 = int(y1 * 0.6)   # Top of ROI
    
    # Calculate x values based on the line equation
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [x1, y1, x2, y2]

def draw_lanes(frame, left_lane, right_lane):
    """
    Draw the detected lanes on the frame.
    """
    line_image = np.zeros_like(frame)
    
    if left_lane is not None:
        x1, y1, x2, y2 = left_lane
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    if right_lane is not None:
        x1, y1, x2, y2 = right_lane
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine the line image with the original frame
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def check_lane_departure(frame, left_lane, right_lane):
    """
    Check if the vehicle is departing from its lane.
    """
    height, width = frame.shape[:2]
    
    # If we can't detect both lanes, can't determine departure
    if left_lane is None or right_lane is None:
        return frame, False
    
    # Calculate the center of the detected lane
    bottom_y = height - 1
    left_x = left_lane[0]  # Bottom x of left lane
    right_x = right_lane[0]  # Bottom x of right lane
    
    lane_center = (left_x + right_x) // 2
    frame_center = width // 2
    
    # Calculate deviation from center
    deviation = abs(lane_center - frame_center) / (width / 2)
    
    # If deviation is too high, trigger warning
    if deviation > 0.3:  # Adjust threshold as needed
        cv2.putText(frame, "LANE DEPARTURE WARNING!", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        return frame, True
    
    return frame, False

def process_video(video_path):
    """
    Process a video file for lane detection.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Define the codec and create VideoWriter object
    output_path = output_dir / f"output_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        # Process the frame
        edges = preprocess_frame(frame)
        roi = region_of_interest(edges)
        
        left_lines, right_lines = detect_lane_lines(roi)
        
        # Get the average line for left and right lanes
        left_lane = average_slope_intercept(left_lines, frame.shape) if left_lines else None
        right_lane = average_slope_intercept(right_lines, frame.shape) if right_lines else None
        
        # Draw the lanes
        frame_with_lanes = draw_lanes(frame, left_lane, right_lane)
        
        # Check for lane departure
        frame_with_warning, _ = check_lane_departure(frame_with_lanes, left_lane, right_lane)
        
        # Display the frame
        cv2.imshow('Lane Detection', frame_with_warning)
        
        # Write the frame to output video
        out.write(frame_with_warning)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Check if video file exists
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Please place your video file in the same directory as this script.")
        print("You can download sample videos from:")
        print("- https://www.kaggle.com/datasets/andrewmvd/road-segmentation")
        print("- https://www.kaggle.com/datasets/brsdincer/lane-detection")
        return
    
    print("Starting lane detection...")
    print("Press 'q' to quit.")
    
    process_video(video_path)

if __name__ == "__main__":
    main()
