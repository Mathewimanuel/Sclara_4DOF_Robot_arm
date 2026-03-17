"""
Camera Calibration for 3D ArUco Tracking
Calibrates camera to get accurate depth and position measurements
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
import time


class CameraCalibrator:
    """
    Handles camera calibration using checkerboard pattern
    """
    
    def __init__(self, checkerboard_size=(9, 6), square_size_mm=25.0):
        """
        Initialize calibrator
        
        Args:
            checkerboard_size: (columns, rows) of INNER corners
            square_size_mm: Size of each square in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        
        # Prepare object points (3D points in real world space)
        # Points are at Z=0 (checkerboard is flat)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), 
                            np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm  # Convert to mm
        
        # Storage for calibration
        self.obj_points = []  # 3D points in real world
        self.img_points = []  # 2D points in image
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        self.reprojection_error = None
    
    def capture_calibration_images(self, num_images=15, save_dir='calibration_images'):
        """
        Interactive calibration image capture
        
        Args:
            num_images: Target number of calibration images
            save_dir: Directory to save captured images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return False
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
        
        print("="*70)
        print("CAMERA CALIBRATION - Image Capture")
        print("="*70)
        print()
        print(f"Checkerboard pattern: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} inner corners")
        print(f"Square size: {self.square_size_mm}mm")
        print()
        print("INSTRUCTIONS:")
        print("  1. Print the checkerboard pattern (will show you link)")
        print("  2. Mount on flat, rigid surface (cardboard, clipboard)")
        print("  3. Hold pattern in different positions and angles")
        print("  4. When corners detected (GREEN), press SPACE to capture")
        print(f"  5. Capture {num_images} images from various angles")
        print()
        print("TIPS FOR GOOD CALIBRATION:")
        print("  • Fill frame with checkerboard (get close)")
        print("  • Tilt at different angles (not just flat)")
        print("  • Move to different areas of camera view")
        print("  • Keep pattern flat (no bending!)")
        print("  • Ensure good lighting")
        print()
        print("CONTROLS:")
        print("  SPACE - Capture image (when corners detected)")
        print("  ESC   - Finish early")
        print("  'q'   - Quit")
        print()
        print("="*70)
        print()
        
        captured = 0
        last_capture_time = 0
        
        try:
            while captured < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to read frame")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret_corners, corners = cv2.findChessboardCorners(
                    gray, self.checkerboard_size, None
                )
                
                display = frame.copy()
                
                # Draw status overlay
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
                
                if ret_corners:
                    # Refine corner positions for better accuracy
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                               30, 0.001)
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    
                    # Draw corners
                    cv2.drawChessboardCorners(
                        display, self.checkerboard_size, corners_refined, ret_corners
                    )
                    
                    # Status - READY TO CAPTURE
                    status = f"DETECTED - Press SPACE to capture ({captured}/{num_images})"
                    color = (0, 255, 0)
                    
                    # Prevent too-rapid captures
                    current_time = time.time()
                    if current_time - last_capture_time < 0.5:
                        status += " [Wait...]"
                        color = (0, 255, 255)
                
                else:
                    # Status - NOT DETECTED
                    status = f"Move checkerboard into view ({captured}/{num_images})"
                    color = (0, 0, 255)
                    corners_refined = None
                
                # Display status
                cv2.putText(display, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Progress bar
                progress_width = int((captured / num_images) * (display.shape[1] - 20))
                cv2.rectangle(display, (10, 50), (10 + progress_width, 65), 
                             (0, 255, 0), -1)
                cv2.rectangle(display, (10, 50), (display.shape[1] - 10, 65), 
                             (100, 100, 100), 2)
                
                cv2.imshow('Camera Calibration - Capture', display)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Handle keyboard
                if key == ord(' ') and ret_corners and corners_refined is not None:
                    # Capture image
                    current_time = time.time()
                    if current_time - last_capture_time >= 0.5:  # Debounce
                        img_path = os.path.join(save_dir, f'calib_{captured:02d}.jpg')
                        cv2.imwrite(img_path, frame)
                        
                        self.obj_points.append(self.objp)
                        self.img_points.append(corners_refined)
                        
                        captured += 1
                        last_capture_time = current_time
                        
                        print(f"✓ Captured {captured}/{num_images}")
                
                elif key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\nCapture stopped by user")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print()
        if captured >= 10:
            print(f"✓ Captured {captured} images - Ready to calibrate!")
            return True
        else:
            print(f"✗ Only captured {captured} images (need at least 10)")
            return False
    
    def calibrate(self, image_shape=None):
        """
        Perform camera calibration from captured images
        
        Args:
            image_shape: (width, height) or None to use saved images
        """
        if len(self.obj_points) < 10:
            print("ERROR: Need at least 10 calibration images")
            return False
        
        # Use first image to get shape if not provided
        if image_shape is None:
            # Assume all images same size, get from first captured
            image_shape = (640, 480)  # Default
        
        print()
        print("="*70)
        print("PERFORMING CALIBRATION")
        print("="*70)
        print(f"Using {len(self.obj_points)} images...")
        print("This may take 10-30 seconds...")
        print()
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, 
            self.img_points, 
            image_shape,
            None, 
            None
        )
        
        if not ret:
            print("✗ Calibration failed!")
            return False
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibrated = True
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
        
        self.reprojection_error = total_error / len(self.obj_points)
        
        print("="*70)
        print("✓ CALIBRATION SUCCESSFUL!")
        print("="*70)
        print()
        print(f"Reprojection Error: {self.reprojection_error:.3f} pixels")
        if self.reprojection_error < 0.5:
            print("  → EXCELLENT calibration!")
        elif self.reprojection_error < 1.0:
            print("  → GOOD calibration")
        else:
            print("  → FAIR calibration (consider recapturing)")
        print()
        print("Camera Matrix:")
        print(self.camera_matrix)
        print()
        print("Distortion Coefficients:")
        print(self.dist_coeffs)
        print()
        
        return True
    
    def save_calibration(self, filename='camera_calibration.pkl'):
        """Save calibration to file"""
        if not self.calibrated:
            print("ERROR: Camera not calibrated yet")
            return False
        
        calib_data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'reprojection_error': self.reprojection_error,
            'checkerboard_size': self.checkerboard_size,
            'square_size_mm': self.square_size_mm,
            'image_shape': (640, 480)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(calib_data, f)
        
        print(f"✓ Calibration saved to: {filename}")
        print()
        return True
    
    def load_calibration(self, filename='camera_calibration.pkl'):
        """Load calibration from file"""
        if not os.path.exists(filename):
            print(f"ERROR: Calibration file not found: {filename}")
            return False
        
        with open(filename, 'rb') as f:
            calib_data = pickle.load(f)
        
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeffs']
        self.reprojection_error = calib_data.get('reprojection_error', None)
        self.calibrated = True
        
        print(f"✓ Calibration loaded from: {filename}")
        print(f"  Reprojection error: {self.reprojection_error:.3f} pixels")
        print()
        
        return True


def print_checkerboard_instructions():
    """Print instructions for getting checkerboard pattern"""
    print()
    print("="*70)
    print("CHECKERBOARD PATTERN NEEDED")
    print("="*70)
    print()
    print("You need a 9x6 checkerboard pattern (inner corners).")
    print()
    print("OPTION 1 - Download and print:")
    print("  https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png")
    print()
    print("OPTION 2 - Generate online:")
    print("  https://calib.io/pages/camera-calibration-pattern-generator")
    print("  Settings: 9 columns, 6 rows, 25mm squares")
    print()
    print("PRINTING INSTRUCTIONS:")
    print("  • Print at ACTUAL SIZE (no scaling)")
    print("  • Use thick paper or cardstock")
    print("  • Mount on flat surface (clipboard, cardboard)")
    print("  • Pattern must be flat (no curves or bends)")
    print()
    print("="*70)
    print()


def main():
    """Run camera calibration workflow"""
    
    print()
    print("="*70)
    print("CAMERA CALIBRATION FOR ROBOT CONTROL")
    print("="*70)
    print()
    print("This will calibrate your camera for accurate 3D position tracking.")
    print("One-time setup - takes about 10 minutes.")
    print()
    
    calibrator = CameraCalibrator(
        checkerboard_size=(9, 6),  # Inner corners
        square_size_mm=25.0        # 25mm squares
    )
    
    # Check if calibration already exists
    if os.path.exists('camera_calibration.pkl'):
        print("Existing calibration file found!")
        choice = input("Use existing calibration? (y/n): ").strip().lower()
        if choice == 'y':
            if calibrator.load_calibration('camera_calibration.pkl'):
                print("✓ Using existing calibration")
                return
    
    # Show checkerboard instructions
    print_checkerboard_instructions()
    
    input("Press ENTER when you have the checkerboard pattern ready...")
    print()
    
    # Capture calibration images
    print("Starting image capture...")
    print()
    
    if not calibrator.capture_calibration_images(num_images=15):
        print()
        print("✗ Calibration aborted - not enough images captured")
        return
    
    # Perform calibration
    if not calibrator.calibrate():
        print("✗ Calibration failed")
        return
    
    # Save calibration
    calibrator.save_calibration('camera_calibration.pkl')
    
    print("="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Use this calibration for 3D ArUco tracking")
    print("  2. Get accurate depth (Z) measurements")
    print("  3. Map hand position to robot workspace")
    print()
    print("The calibration file 'camera_calibration.pkl' will be used")
    print("automatically by the tracking system.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
