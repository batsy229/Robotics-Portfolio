# Autonomous Drone Landing Using ArUco Markers  
**Date:** Aug 2025  

---

## Project Overview  
This project developed a vision-based precision landing system for drones using ArUco markers. The system enables autonomous landing on predefined platforms by integrating real-time marker detection, pose estimation, and drone control. The approach ensures accurate trajectory adjustments during descent for safe and reliable landings.  

---

## Key Features  
- **Autonomous Precision Landing:** Drone detects ArUco marker and aligns itself for landing automatically.  
- **Real-Time Pose Estimation:** Computes translation (x, y, z) and rotation (roll, pitch, yaw) vectors relative to the camera.  
- **Dynamic Trajectory Adjustment:** Drone roll, pitch, yaw, and descent rate updated in real-time using MAVLink and DroneKit-Python.  
- **Safety Protocols:** Return-to-Launch (RTL) and geofence implemented to ensure safe operations in outdoor environments.  
- **Camera Calibration & Frame Alignment:** Ensures accurate pose estimation by correcting for intrinsic/extrinsic camera parameters and aligning camera and drone frames.  

---

## Technologies & Tools  
**Programming Languages:** Python  
**Robotics Frameworks:** DroneKit-Python, MAVLink  
**Computer Vision:** OpenCV ArUco library, Camera Calibration  
**Hardware:** Holybro X650 Drone Kit, Pixhawk 6C, Raspberry Pi 4, Pi HQ Camera, M10 GPS, RC Controller & Receiver  

---

## Project Outcomes  
- Achieved autonomous drone landing on a predefined platform using a single ArUco marker.  
- Successfully streamed pose data from Raspberry Pi to Pixhawk for dynamic flight adjustments.  
- Implemented safe descent with controlled altitude and soft landing at 50 cm threshold.  
- Demonstrated robust performance under moderate outdoor conditions with adaptive thresholding and search-and-realign for marker detection.  

---

## Repository Contents  
- `/src` – Source code for marker detection, pose estimation, and drone control modules.  
- `/reports` – Project report and documentation.  

