# Object Tracking & Policy Monitoring System

A lightweight object tracking and video analysis system built using YOLO and ByteTrack. The solution is designed for execution with minimal computational requirements, making it suitable for deployment on CPU-only devices.


# 📌 Features

 - Low Computational Footprint: Optimized to allow efficient video processing even on low-power hardware.

 - Line Crossing Detection: Supports policy line crossing checks to monitor object movement across user defined virtual boundaries.


# Execution
 1. **Local Setup**
   
    Install the dependencies:

    ```Bash
    pip install -r requirements.txt
    ```

    Run the main processing script:

    ```Bash
    python3 main.py
    ```

 2. **Docker Execution**
   
    You can alternatively run the application using Docker environment:

    ```Bash
    docker compose run object-tracker
    ```


# Configuration

All parameters and configurations are managed via the `configs.yaml` file.

 - **Input/Output:** Specify input video paths and output file names in `configs.yaml`.
 - **Policy Lines:** Define monitoring boundaries in `./data/policy.json`.


# Demo

<video src="https://github.com/user-attachments/assets/a4bff196-7364-4847-a0cf-fb7b0b9956d7" 
       loop 
       muted 
       autoplay 
       playsinline 
       style="max-width: 100%;">
</video>