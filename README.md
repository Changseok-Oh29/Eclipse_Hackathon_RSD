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
## Terminal 2 (현재는 비활) -> kuksa 실행
cd ~/Eclipse_Hackathon_RSD/Eclipse/sensor  # 파일 경로 진입

docker run -it --rm \ # 도커 실행 
  -p 55555:55555 \
  -v $(pwd)/myvss.json:/app/vss.json \
  ghcr.io/eclipse/kuksa.val/databroker:latest \
  --vss /app/vss.json


        
## perception.py 실행
항상 먼저 실행
    
    python3 perception.py
## decision.py 실행
perception 실행 후 약 2-3초 기다렸다가 실행 
    
    python3 decision.py

    
## 데이터 브로커 확인 
docker run -it --rm --net=host ghcr.io/eclipse-kuksa/kuksa-databroker-cli:latest # CLI실행



ACC (레이다 기반)
subscribe Vehicle.ADAS.ACC.Distance
subscribe Vehicle.ADAS.ACC.RelSpeed
subscribe Vehicle.ADAS.ACC.TTC
subscribe Vehicle.ADAS.ACC.HasTarget
subscribe Vehicle.ADAS.ACC.LeadSpeedEst

LK (차선 유지)
subscribe Vehicle.ADAS.LK.Steering

Road (카메라 기반 노면 분석)
subscribe Vehicle.Private.Road.State
subscribe Vehicle.Private.Road.Confidence
subscribe Vehicle.Private.Road.Ts
subscribe Vehicle.Private.Road.Metrics.SRI
subscribe Vehicle.Private.Road.Metrics.SRI_rel
subscribe Vehicle.Private.Road.Metrics.ED

Slip (슬립 추정)
subscribe Vehicle.Private.Slip.State
subscribe Vehicle.Private.Slip.Quality
subscribe Vehicle.Private.Slip.Confidence
subscribe Vehicle.Private.Slip.Ts
subscribe Vehicle.Private.Slip.Metrics.v
subscribe Vehicle.Private.Slip.Metrics.vx
subscribe Vehicle.Private.Slip.Metrics.vy
subscribe Vehicle.Private.Slip.Metrics.alpha_deg
subscribe Vehicle.Private.Slip.Metrics.ax_mean
subscribe Vehicle.Private.Slip.Metrics.ay_abs_mean
subscribe Vehicle.Private.Slip.Metrics.long_residual
subscribe Vehicle.Private.Slip.Metrics.wheel_slip_mean
subscribe Vehicle.Private.Slip.Metrics.wheel_odo_v_mean
subscribe Vehicle.Private.Slip.Metrics.kappa_est

StateFused (카메라+슬립 융합)
subscribe Vehicle.Private.StateFused.State
subscribe Vehicle.Private.StateFused.Confidence
subscribe Vehicle.Private.StateFused.Ts
subscribe Vehicle.Private.StateFused.Metrics.W_cam
subscribe Vehicle.Private.StateFused.Metrics.W_slip
subscribe Vehicle.Private.StateFused.Metrics.LatencyMs














    
