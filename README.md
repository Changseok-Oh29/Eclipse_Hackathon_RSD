# RSD - Ready Set Drive
## Challenge : Virtual SDV Lab Challenge
## Concept
Our system leverages Eclipse SDV tools to integrate ACC (Adaptive Cruise Control), LKAS (Lane Keeping Assist System), and real-time road condition detection. The goal is to enable safe distance management with lead vehicles and ensure overall driving safety.
## Architecture Overview
### 1. Sensor Layer (CARLA Simulator)
Camera: Lane detection and road condition estimation (ROI extraction)
Lidar: Distance and relative speed to lead vehicle
### 2. Data Transport Layer
Zenoh: Publishes high-bandwidth ROI/image data using zero-copy transport
Kuksa VSS: Exchanges numeric vehicle signals such as speed, lane center offset, distance, TTC, and control commands using the VSS model
### 3. Perception Layer
Processes ROI from camera input
Performs lane detection
Calculates distance and relative speed from lidar
Estimates road condition (dry / wet / icy)
Publishes results via Kuksa VSS and Zenoh
### 4. Decision Layer
LKAS: Keeps the vehicle centered in the lane through steering control
ACC: Cruise, follow, and braking control based on lead vehicle distance and speed
AEB: Emergency braking triggered by TTC thresholds
Road condition input adjusts braking and acceleration sensitivity
### 5. Control Layer
Sends control commands (Throttle / Brake / Steer) via Kuksa VSS
Applies control commands to the ego vehicle in the CARLA simulator
### 6. Orchestration Layer
Ankaios: Manages deployment and orchestration of Perception, Decision, and Control modules

# 
Eclipse 폴더 안에 perception, decision 파일 있습니다. 
각자 수정하실 부분 수정하시고 commit, push해주세요.
push하실 때 어떤 부분 수정하셨는지 description 자세하게 설명 부탁드립니다.
#

# Command
## Terminal 1
    prime-run ./CarlaUE4.sh

    ## CLI 창에서 ##
    Open /Game/Carla/Maps/Town04
## Terminal 2 (현재는 비활)
    podman run --rm --name kuksa-db --net=host \
        -v "$PWD/model.fixed.json":/data/model.json:ro \
        ghcr.io/eclipse-kuksa/kuksa-databroker:latest \
        --insecure --vss /data/model.json
## perception.py 실행
    python3 perception.py
## decision.py 실행
    python3 decision.py
