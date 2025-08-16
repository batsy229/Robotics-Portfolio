import numpy as np
import cv2
import cv2.aruco as aruco
import time
import math
from picamera2 import Picamera2
import pickle
from pymavlink import mavutil

### 1. Initialize MAVLink Connection
pixhawk = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
pixhawk.wait_heartbeat()  # Confirm connection

### 2. Load Camera Calibration Parameters
with open('/home/goku/python/cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)
with open('/home/goku/python/dist.pkl', 'rb') as f:
    camera_distortion = pickle.load(f)

### 3. Define Drone Height (Adjust Landing)
DRONE_HEIGHT_CM = 25  # Distance between camera and landing gear


###  4. Define ArUco Tracking Class
class ArucoSingleTracker():
    def __init__(self, marker_id, marker_size):
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.is_detected = False

        # Initialize Picamera2 (but don’t start it yet)
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (1280, 720), "format": "RGB888"})  # Optimized resolution
        self.picam2.configure(config)

        # Define ArUco dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters()

    def track_marker(self, max_wait_time=10):
        """Detects ArUco marker after camera activation."""
        self.picam2.start()  # Camera ON only after RTL!
        time.sleep(2)  # Camera stabilization delay

        start_time = time.time()
        marker_found = False

        while time.time() - start_time < max_wait_time:
            frame = self.picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None and self.marker_id in ids:
                marker_found = True

                # Estimate marker pose
                ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, camera_matrix, camera_distortion)
                rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                x, y, z = tvec[0], tvec[1], tvec[2]

                #  Adjust Z to Account for Camera Height
                adjusted_z = max(z - DRONE_HEIGHT_CM, 0)  # Prevent negative values

                print(f"Marker Found at: x={x:.1f} y={y:.1f} z={z:.1f} (Adjusted Z: {adjusted_z:.1f})")

                #  Ensure Safe Landing Decision
                if adjusted_z <= 5:  # If near ground, trigger standard landing
                    print("Drone close to ground—landing now.")
                    command_land()
                else:
                    send_pose(x, y, adjusted_z, rvec)

                break  # Stop tracking after reliable detection

            time.sleep(0.1)  # Prevent excessive processing

        self.picam2.stop()

        return marker_found  # Return result to decide next step


###  5. Send Pose to Pixhawk
def send_pose(x, y, adjusted_z, rvec):
    """Adjusts landing height by subtracting drone height."""
    pixhawk.mav.set_position_target_global_int_send(
        0,  # System ID
        0,  # Component ID
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b110111111000,  # Enable XYZ + Yaw control
        x, y, -adjusted_z,  # Adjusted Z position
        0, 0, 0,  # Velocity
        0, 0, 0,  # Acceleration
        rvec[2], 0  # Yaw
    )


###  6. Monitor RTL & Manage Descent Strategy
def monitor_rtl():
    """Waits for RTL, starts ArUco tracking, and manages fallback descent."""
    while True:
        msg = pixhawk.recv_match(type='HEARTBEAT', blocking=True)
        if msg.custom_mode == mavutil.mavlink.MAV_MODE_AUTO_RTL:
            print("RTL activated! Turning on camera and tracking marker...")

            aruco_tracker = ArucoSingleTracker(marker_id=69, marker_size=15)
            marker_found = aruco_tracker.track_marker(max_wait_time=10)

            if not marker_found:
                print("Marker not detected within timeout, continuing RTL descent...")
                continue_standard_rtl_descent()

            break  # Stop monitoring once tracking is active

        time.sleep(0.5)  # Monitor RTL at intervals


###  7. Continue Normal RTL Descent if Marker Not Found
def continue_standard_rtl_descent():
    """Allows normal RTL descent if marker isn't detected."""
    print("Continuing normal RTL descent...")
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_NAV_CONTINUE_AND_CHANGE_ALT,
        0, 0, 0, 0, 0, 0, 0
    )


###  8. Command Pixhawk to Land
def command_land():
    """Triggers landing after successful marker detection."""
    print("Valid marker detected, executing landing...")
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0
    )


### 9. Run Full System
monitor_rtl()