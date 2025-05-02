# Multifunctional Drone Control System

This repository contains the full Python implementation for a multifunctional drone control system designed for the Bitcraze Crazyflie 2.1 drone. The system integrates four key components: hand gesture recognition, autonomous obstacle avoidance, 3D flight path and obstacle mapping, and first-person video (FPV) streaming via the AI-Deck. Together, these features enable intuitive and safe drone navigation, making it accessible for beginners and useful for research in human-drone interaction and autonomous systems.

## Description

This project showcases a fully integrated drone control system that combines hand-gesture recognition, autonomous obstacle avoidance, 3D environment mapping, and FPV video streaming, all tailored for the Bitcraze Crazyflie 2.1 drone platform. Using Google’s MediaPipe Hands framework, the system detects and classifies predefined hand gestures from a webcam feed, which are then mapped to drone flight commands. The Multi-Ranger Deck enables the drone to detect obstacles in all horizontal directions using infrared time-of-flight sensors and autonomously react by backing away from nearby objects, preventing collisions without user intervention. Meanwhile, the Flow Deck V2 provides accurate position tracking, and all flight data is used to generate an interactive 3D map of the drone’s path and the locations of detected obstacles using Plotly. Additionally, the AI-Deck streams either grayscale or color video back to the user, allowing for FPV navigation even when the drone moves beyond direct line-of-sight. This system offers a safer, more intuitive approach to piloting microdrones—ideal for educational use, human-drone interaction research, or experimental robotics projects.

## Getting Started

### Dependencies
#### Operating System
* Windows or MacOS
#### Hardware Requirements
* Bitcraze Crazyflie 2.1 Drone
* Flow Deck V2
* Multi-Ranger Deck
* AI-Deck 1.1
* Crazyradio Dongle
#### Python Version
* Python 3.8+ (recommended)
#### Python Libraries
Install the following libraries using pip:
```
pip install opencv-python
pip install mediapipe
pip install pyautogui
pip install plotly
pip install pandas
pip install numpy
```
#### Bitcraze Python Libraries
Install the Bitcraze Crazyflie Python API:
```
pip install cflib
```
#### Tools for AI-Deck Setup
* Ubuntu Linux
* GAP8 toolchain and SDK from Bitcraze
* Python sockets for communication
* OpenCV (for video decoding and display)

### Installing

#### 1. Clone the Repository
```
git clone https://github.com/PING-ResearchLab/DroneLab-Code.git
cd DroneLab-Code
```
#### 2. Proper Files
Ensure the following files are available:
* AiDeckMapping.py: Contains entire multifunctional drone system
* util.py: Contains helper functions used for gesture angle/distance calculations
#### 3. Configure AI-Deck (Linux Only)
* Follow Step-by-Step Process (see https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/)

### Executing program

Follow these steps to run the full system, which includes gesture control, obstacle avoidance, 3D mapping, and FPV streaming with the AI-Deck:
#### 1. Power On the Crazyflie 2.1 Drone
* Ensure the Flow Deck V2, Multi-Ranger Deck, and AI-Deck are properly mounted
* Connect the Crazyradio USB dongle to your computer
#### 2. Run the AiDeckMapping Python Script
This script:
* Initializes the Crazyflie connection
* Starts gesture recognition with your webcam
* Activates obstacle detection
* Begins AI-Deck video streaming
* Starts logging flight data for the 3D map
#### 3. Use Hand Gestures to Fly
* Position your hand about 1 foot from the webcam
* Use predefined gestures (e.g., point, peace, fist) to control the drone
#### 4. Land the Drone
* Use the “OK” gesture to land the drone safely and end the flight session
#### 5. View Output
* crazyflie_3d_path.html: Interactive 3D map of flight path and obstacles

## Authors

Contributors names and contact info

* Ben Taylor (https://github.com/btaylor49787)
* Pingping Zhu (PING-ResearchLab)
