#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Env_acc_kuksa.py — CARLA 레이더 → Kuksa(ACC 센서 5종), 카메라 → Zenoh
- ACC 센서:
  Vehicle.ADAS.ACC.Distance (m)
  Vehicle.ADAS.ACC.RelSpeed (m/s, +: 접근)
  Vehicle.ADAS.ACC.TTC (s, 열림이면 큰 수 9999.9)
  Vehicle.ADAS.ACC.HasTarget (bool)
  Vehicle.ADAS.ACC.LeadSpeedEst (m/s)
"""

import os, sys, time, json, math, argparse
import numpy as np
import cv2
import carla
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ---------------- VSS 경로 ----------------
ACC_ROOT = "Vehicle.ADAS.ACC"
VSS = {
    "Distance":     f"{ACC_ROOT}.Distance",
    "RelSpeed":     f"{ACC_ROOT}.RelSpeed",
    "TTC":          f"{ACC_ROOT}.TTC",
    "HasTarget":    f"{ACC_ROOT}.HasTarget",
    "LeadSpeedEst": f"{ACC_ROOT}.LeadSpeedEst",
}

# ---------------- 유틸 ----------------
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def make_zenoh_session(endpoint: str):
    cfg = zenoh.Config()
    try:
        cfg.insert_json5("mode", '"client"')
        cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    except AttributeError:
        cfg.insert_json("mode", '"client"')
        cfg.insert_json("connect/endpoints", f'["{endpoint}"]')
    return zenoh.open(cfg)

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser("Env_acc_kuksa")
    # CARLA
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--fps',  type=int, default=40)
    ap.add_argument('--spawn_idx', type=int, default=328)
    # 카메라
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=1)
    # Zenoh
    ap.add_argument('--zenoh_endpoint', default='tcp/127.0.0.1:7447')
    ap.add_argument('--topic_cam', default='carla/cam/front')
    # Kuksa
    ap.add_argument('--kuksa_host', default='127.0.0.1')
    ap.add_argument('--kuksa_port', type=int, default=55555)
    args = ap.parse_args()

    # -------- Zenoh --------
    z = make_zenoh_session(args.zenoh_endpoint)
    pub_cam = z.declare_publisher(args.topic_cam)

    # -------- Kuksa --------
    kc = VSSClient(args.kuksa_host, args.kuksa_port)
    kc.connect()
    print(f"[KUKSA] connected @ {args.kuksa_host}:{args.kuksa_port}")

    # -------- CARLA 연결/세팅 --------
    client = carla.Client(args.host, args.port); client.set_timeout(5.0)
    world = client.get_world()
    dt = 1.0 / max(1, args.fps)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.substepping = True
    settings.max_substep_delta_time = 0.005
    settings.max_substeps = 10
    world.apply_settings(settings)
    print(f"[WORLD] sync=True dt={dt:.3f}s substepping=on")

    # -------- 스폰 --------
    bp = world.get_blueprint_library()
    sps = world.get_map().get_spawn_points()
    if not sps:
        raise RuntimeError("No spawn points in map")

    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', 'ego')
    tf_ego = sps[min(max(0, args.spawn_idx), len(sps)-1)]
    ego = world.try_spawn_actor(ego_bp, tf_ego)
    if ego is None:
        raise RuntimeError("Failed to spawn Ego")

    lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
    lead_bp.set_attribute('role_name', 'lead')
    lead_wp = world.get_map().get_waypoint(tf_ego.location).next(30.0)[0]
    tf_lead = lead_wp.transform; tf_lead.location.z = tf_ego.location.z
    lead = world.try_spawn_actor(lead_bp, tf_lead)
    if lead is None:
        raise RuntimeError("Failed to spawn Lead")
    print(f"[SPAWN] ego={ego.id}, lead={lead.id}")

    # -------- 카메라 --------
    latest_front = {'bgr': None}
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    def on_cam(img: carla.Image):
        # 원본 BGRA를 그대로 Zenoh로, 메타 첨부
        buf = memoryview(img.raw_data)
        att = json.dumps({
            "w": img.width, "h": img.height, "c": 4, "format": "bgra8",
            "stride": img.width * 4, "frame": int(img.frame),
            "sim_ts": float(img.timestamp), "pub_ts": time.time()
        }).encode("utf-8")
        pub_cam.put(bytes(buf), attachment=att)

        # 디스플레이용 BGR
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest_front['bgr'] = arr[:, :, :3].copy()

    cam.listen(on_cam)

    # -------- 레이더 --------
    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120')
    radar_bp.set_attribute('horizontal_fov', '20')
    radar_bp.set_attribute('vertical_fov', '10')
    radar_bp.set_attribute('sensor_tick', str(dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)

    last_pub_ts = 0.0
    min_period = dt  # 레이트리밋: 센서 틱과 동일

    def on_radar(meas: carla.RadarMeasurement):
        nonlocal last_pub_ts
        now = time.time()
        if (now - last_pub_ts) < min_period:
            return

        if len(meas) == 0:
            kc.set_current_values({ VSS["HasTarget"]: Datapoint(False) })
            last_pub_ts = now
            return

        # 가장 가까운 타겟 하나만 사용(데모용)
        best = min(meas, key=lambda d: d.depth)
        distance = float(best.depth)             # m
        if not math.isfinite(distance):
            return

        # CARLA: velocity + = 멀어짐 / - = 접근 → ACC 표준(접근 +)으로 변환
        rel_speed_acc = float(-best.velocity)    # m/s

        # 간단한 선도차 추정: ego 속도 - rel_speed
        v = ego.get_velocity()
        ego_speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)  # m/s
        lead_speed_est = clamp(ego_speed - rel_speed_acc, 0.0, 100.0)

        # TTC (접근일 때만 유한)
        if rel_speed_acc > 0.0:
            ttc = max(0.1, distance / rel_speed_acc)
        else:
            ttc = float('inf')

        # 범위 정리
        distance = clamp(distance, 0.0, 500.0)
        rel_speed_acc = clamp(rel_speed_acc, -100.0, 100.0)
        ttc_val = 9999.9 if not math.isfinite(ttc) else clamp(ttc, 0.0, 1e4)

        kc.set_current_values({
            VSS["Distance"]:     Datapoint(distance),
            VSS["RelSpeed"]:     Datapoint(rel_speed_acc),
            VSS["TTC"]:          Datapoint(ttc_val),
            VSS["HasTarget"]:    Datapoint(True),
            VSS["LeadSpeedEst"]: Datapoint(lead_speed_est),
        })
        last_pub_ts = now

    radar.listen(on_radar)

    # -------- 표시창 --------
    if args.display:
        cv2.namedWindow('front', cv2.WINDOW_NORMAL)

    print("[RUN] Env streaming... (Ctrl+C to stop)")
    last_status = time.time()
    try:
        while True:
            world.tick()

            # 상태 로그
            now = time.time()
            if now - last_status >= 5.0:
                v = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
                loc = ego.get_transform().location
                print(f"[STATUS] ego@({loc.x:.1f},{loc.y:.1f}) {speed_kmh:.1f} km/h")
                last_status = now

            if args.display and latest_front['bgr'] is not None:
                cv2.imshow('front', latest_front['bgr'])
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            time.sleep(0.002)

    except KeyboardInterrupt:
        pass
    finally:
        try: cam.stop(); radar.stop()
        except: pass
        for a in [cam, radar, lead, ego]:
            try: a.destroy()
            except: pass
        try: pub_cam.undeclare(); z.close()
        except: pass
        try: kc.disconnect()
        except: pass
        world.apply_settings(original_settings)
        if args.display:
            try: cv2.destroyAllWindows()
            except: pass
        print("[CLEAN] Env done.")

if __name__ == "__main__":
    main()
