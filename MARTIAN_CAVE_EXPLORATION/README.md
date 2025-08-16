# Martian Cave Exploration (Simulation Project)

**Date:** November 2024  
**Tools & Technologies:** ROS, RGB-D Cameras, LiDAR, YOLOv11, Python, Gazebo, RViz, SLAM

---

## Project Overview
This project simulates autonomous exploration and mapping of a Martian cave using ROS and multi-sensor fusion. The robot autonomously explores unknown environments, detects artifacts using computer vision, and estimates their positions in a 3D map. The system integrates **SLAM (Simultaneous Localization and Mapping)**, motion planning, and artifact perception for end-to-end autonomous exploration.

---

## Key Features

### 1. Autonomous Exploration (Frontier-Based Planning)
- Implemented a **frontier-based exploration algorithm** for efficient cave mapping.
- Robot starts without prior knowledge of the environment and navigates autonomously to discover unknown areas.
- Decision-making logic prioritizes unexplored regions to maximize map coverage.
- Planning sequence:
  - Random/pre-specified movements as a baseline.
  - Custom `PlannerType` integrated into `main_loop`.
  - Iterative improvements to exploration performance.

### 2. Close-Range Inspection
- Robot detects specific artifacts using YOLOv11.
- Stops exploration when an artifact is detected.
- Approaches the artifact to inspect it closely using depth camera data.
- Ensures the camera captures the artifact at an optimal distance for analysis.

### 3. Artifact Localization and Display
- Estimated artifact locations in world coordinates using depth camera and multi-detection averaging.
- Bounding boxes displayed in camera feed to indicate artifact direction.
- RViz visualization of artifact positions using **Markers** for clarity.

---

## Results
- Autonomous mapping of simulated Martian cave environments.
- Artifact detection accuracy: **92%** on 1,372 custom images.
- Efficient exploration with frontier-based strategy.
- Accurate artifact localization and visualization in RViz.

---

## Video Links
https://drive.google.com/drive/folders/1HyhIoH4_nZaupnH7gsl6iiaBAddTJo-m?usp=drive_link

