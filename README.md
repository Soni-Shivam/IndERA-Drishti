# IndERA-Drishti: Vision-Language Integrated Robotic Sorting for Industry 4.0
<img width="718" height="797" alt="image" src="https://github.com/user-attachments/assets/86608069-8dc6-40dc-96a3-f189ec38b91e" />

Videos for Demo can be found here - https://drive.google.com/drive/folders/1P4b91-Dg2tlPtwPfC3rTbewuPNpMdRDS
Project Report for more details - https://github.com/Soni-Shivam/IndERA-Drishti/blob/main/IndERA-Drishti.pdf
## Overview
IndERA-Drishti is an autonomous sorting system that gives traditional industrial robotic arms cognitive vision using advanced AI. Developed for decentralized sorting of unstructured, unknown objects (such as end-of-life recycling waste), the system leverages Google's PaliGemma Vision-Language Model (VLM) for zero-shot object classification. 

The system autonomously identifies items on a moving conveyor belt, calculates precise spatial coordinates, and executes pick-and-place maneuvers using an iterative Inverse Kinematics (IK) solver.

## Key Features
* **Zero-Shot Perception**: Uses PaliGemma-3b-mix-224 to classify unstructured objects into targeted inventory groups without prior specific training.
* **Robust Inverse Kinematics**: Custom multi-attempt iterative IK solver with position-only masks, joint limit enforcement, and minimal-movement cost functions to ensure safe real-world operation.
* **Full Automation Loop**: Synchronized control between an HC-SR04 ultrasonic sensor, a conveyor belt, and a 6-DOF robotic arm for continuous pick-and-place operations.
* **Digital Twin Synchronization**: Real-time 3D visualization using the Swift simulator, operating in tandem with the physical Arduino-controlled arm.
* **Web Dashboard**: Flask-based interface for real-time ESP32-CAM monitoring, interactive Region of Interest (ROI) cropping, and live inference logs.

## System Architecture
<img width="2048" height="2048" alt="Gemini_Generated_Image_z4abooz4abooz4ab" src="https://github.com/user-attachments/assets/d739cd6c-c5cb-4fde-a56f-5b036a93fba4" />

### Hardware Components
* **Perception Node**: ESP32-CAM module providing overhead live video feed.
* **Robotic Arm**: 6-DOF 3D-printed multi-articulated manipulator (Base, Shoulder, Elbow, Wrist Roll, Wrist Pitch, Gripper).
* **Conveyor System**: Mini conveyor belt driven by a BO motor and L298N/L293D motor driver.
* **Sensors**: HC-SR04 ultrasonic sensor for object detection and conveyor stopping.
* **Low-Level Controller**: Arduino (Mega/Uno) parsing serial commands and controlling servos/motors.
<img width="3000" height="1723" alt="circuit_image (2)" src="https://github.com/user-attachments/assets/202ea06f-7a55-426d-9ad4-25b34e30f58f" />

### Software Stack
* **AI/Vision**: PyTorch, Transformers, PaliGemma VLM, OpenCV, PIL.
* **Kinematics**: roboticstoolbox-python, spatialmath-python, custom URDF.
* **Backend**: Flask, Server-Sent Events (SSE) for live UI updates.
* **Firmware**: Arduino C++ with custom non-blocking serial protocol.

## Repository Structure

* `app.py`: Flask web server serving the dashboard, proxying the ESP32-CAM stream, and triggering inference.
* `model_engine.py`: Offline inference engine managing the PaliGemma model and parsing VLM text outputs into structured data.
* `arm_common.py`: Core robotics module containing the URDF loader, coordinate transformations, and the robust IK solver.
* `robot_controller.py`: Main orchestration script managing the Arduino serial connection, Swift simulator, and the automated conveyor loop.
* `custom_arm.urdf`: XML definition of the robot's physical links, joints, materials, and limits.
* `arm_controller/arm_controller.ino`: Arduino firmware for handling servo movements, conveyor speed, and ultrasonic polling.
* `ik_solver.py` & `ik_visualize.py`: Standalone CLI tools for testing IK calculations and simulating movements without hardware.
* `templates/index.html`: Frontend UI for the camera feed, ROI selection, and system logs.

## Setup & Installation

### 1. Hardware Setup
Upload `arm_controller.ino` to your Arduino. Ensure the servos are connected to pins 3, 5, 6, 9, 10, and 11. Connect the ultrasonic sensor to pins 4 (TRIG) and 2 (ECHO). Connect the conveyor motor driver to pins 7, 8, and 12.

### 2. Environment Configuration
Create a `.env` file in the project root and add your Hugging Face token (required to download the PaliGemma model):
```
HF_TOKEN=your_huggingface_token_here
```

### 3. Python Dependencies
Install the required packages using pip. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
pip install roboticstoolbox-python spatialmath-python swift-sim pyserial
```

## Usage

### Running the Web Dashboard
Start the Flask server to monitor the ESP32-CAM feed and run manual classifications.
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser. You can draw a Region of Interest (ROI) on the camera feed and click "Scan & Classify" to test the VLM.

### Running the Robot Automation
To start the physical robot and the synchronized Swift simulator, run the controller script:
```bash
python robot_controller.py
```
Inside the interactive prompt, you can use manual commands (e.g., `pick 15 0 10`, `home`, `grip:open`) or type `conveyor` to initiate the fully automated sorting loop.

## Affiliation
This system was developed in association with the Digital Enterprise Lab at IIT Bombay.
