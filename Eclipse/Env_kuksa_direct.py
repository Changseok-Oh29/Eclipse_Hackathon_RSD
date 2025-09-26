#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Env_kuksa_direct.py — CARLA 센서(레이더) → Kuksa(VSS 센서경로)에 바로 write
# - 카메라 프레임은 Zenoh publish 그대로 유지(디버그/뷰어용)
# - ACC 필요한 5개 값(distance, rel_speed, ttc, has_target, lead_speed_est)은 Kuksa로 직접 set_current_values

import os, sys, time, json, math, argparse
import numpy as np
import cv2
import carla
import zenoh

# ==== KUKSA 준비 (0.4.0 API) ====
from kuksa_client.grpc import VSSClient, Datapoint

class VSS:
    ROOT = "Vehicle.ADAS.ACC"
    Distance     = f"{ROOT}.Distance"
    RelSpeed     = f"{ROOT}.RelSpeed"
    TTC          = f"{ROOT}.TTC"
    HasTarget    = f"{ROOT}.HasTarget"
    LeadSpeedEst = f"{ROOT}.LeadSpeedEst"

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def clamp(x, lo, hi): return max(lo, min(hi, x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--fps',  type=int, default=20)
    ap.add_argument('--spawn_idx', type=int, default=328)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=1)
    # Kuksa endpoint
    ap.add_argument('--kuksa_host', default='127.0.0.1')
    ap.add_argument('--kuksa_port', type=int, default=55555)
    args = ap.parse_args()

    # === Zenoh(카메라 스트림용) ===
    cfg = zenoh.Config()
    try:
        cfg.insert_json5('mode', '"client"')
        cfg.insert_json5('connect/endpoints', '["tcp/127.0.0.1:7447"]')
    except AttributeError:
        cfg.insert_json('mode', '"client"')
        cfg.insert_json('connect/endpoints', '["tcp/127.0.0.1:7447"]')
    sess = zenoh.open(cfg)
    pub_cam = sess.declare_publisher('carla/cam/front')

    # === Kuksa 연결 ===
    kc = VSSClient(args.kuksa_host, args.kuksa_port)
    kc.connect()
    print(f"[KUKSA] Connected to {args.kuksa_host}:{args.kuksa_port}")

    # === CARLA 연결/세팅 ===
    client = carla.Client(args.host, args.port); client.set_timeout(5.0)
    world = client.get_world()
    dt = 1.0 / max(1, args.fps)
    original_settings = world.get_settings()
    settings = world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=dt; settings.substepping=False
    settings.substepping = True
    settings.max_substep_delta_time = 0.005  # 5 ms
    settings.max_substeps = 10               # 최대 10 서브스텝 (최대 200 Hz 내부 물리)
    world.apply_settings(settings)
    world.apply_settings(settings)
    print(f"[WORLD] synchronous_mode=True, dt={dt:.3f}")

    # === 스폰 ===
    bp = world.get_blueprint_library()
    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', 'ego')
    tf  = world.get_map().get_spawn_points()[max(0, min(args.spawn_idx, len(world.get_map().get_spawn_points())-1))]
    ego = world.try_spawn_actor(ego_bp, tf)
    if ego is None: raise RuntimeError("Failed to spawn Ego.")

    lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
    lead_bp.set_attribute('role_name', 'lead')
    lead_wp = world.get_map().get_waypoint(tf.location).next(30.0)[0]
    lead_tf = lead_wp.transform; lead_tf.location.z = tf.location.z
    lead = world.try_spawn_actor(lead_bp, lead_tf)
    if lead is None: raise RuntimeError("Failed to spawn Lead.")

    # === 센서 ===
    latest_front = {'bgr': None}
    latest_chase = {'bgr': None}

    # camera (front)
    cam_bp = bp.find('sensor.camera.rgb'); cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height)); cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    def on_cam(img: carla.Image):
        buf = memoryview(img.raw_data)
        att = json.dumps({
            "w": img.width, "h": img.height, "c": 4, "format": "bgra8", "stride": img.width*4,
            "frame": int(img.frame), "sim_ts": float(img.timestamp), "pub_ts": time.time()
        }).encode("utf-8")
        pub_cam.put(bytes(buf), attachment=att)
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest_front['bgr'] = arr[:, :, :3].copy()
    cam.listen(on_cam)

    # chase cam (optional)
    chase = None
    try:
        chase_bp = bp.find('sensor.camera.rgb'); chase_bp.set_attribute('image_size_x', str(args.width))
        chase_bp.set_attribute('image_size_y', str(args.height)); chase_bp.set_attribute('fov', '70')
        chase_bp.set_attribute('sensor_tick', str(dt))
        chase_tf = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12.0))
        chase = world.spawn_actor(chase_bp, chase_tf, attach_to=ego)
        def _on_chase(img: carla.Image):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            latest_chase['bgr'] = arr[:, :, :3].copy()
        chase.listen(_on_chase)
    except Exception as e:
        print(f"[WARN] chase attach failed: {e}")

    # radar
    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120'); radar_bp.set_attribute('horizontal_fov', '20')
    radar_bp.set_attribute('vertical_fov', '10'); radar_bp.set_attribute('sensor_tick', str(dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar    = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)

    # === 레이더 → Kuksa(VSS: current) ===
    # CARLA convention: velocity +는 멀어짐, -는 접근 → ACC에선 "접근=+" 쓰고 싶으면 부호 반전
    last_pub_ts = 0.0
    min_period  = dt  # 레이더 tick과 맞춤(20Hz)

    def on_radar(meas: carla.RadarMeasurement):
        nonlocal last_pub_ts
        now = time.time()
        if now - last_pub_ts < min_period:
            return  # rate limit

        if len(meas) == 0:
            # 타깃 없음
            updates = {
                VSS.HasTarget: Datapoint(False),
            }
            kc.set_current_values(updates)
            last_pub_ts = now
            return

        best = min(meas, key=lambda d: d.depth)  # 가장 가까운 물체
        distance = float(best.depth)             # m
        # CARLA vel: + 멀어짐 / - 접근 → 접근=+ 로 변환
        rel_speed_acc = float(-best.velocity)

        # ego 속도 → lead 추정 (간단한 예: ego 속도 계측 후 rel 이용)
        v = ego.get_velocity()
        ego_speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)  # m/s
        lead_speed_est = clamp(ego_speed - rel_speed_acc, 0.0, 100.0)

        # TTC (접근일 때만 유한)
        if rel_speed_acc > 0.0:
            ttc = max(0.1, distance / rel_speed_acc)
        else:
            ttc = float('inf')

        # 클램프/정규화
        distance = clamp(distance, 0.0, 500.0)
        rel_speed_acc = clamp(rel_speed_acc, -100.0, 100.0)
        ttc = (9999.9 if not math.isfinite(ttc) else clamp(ttc, 0.0, 1e4))

        updates = {
            VSS.Distance:     Datapoint(distance),
            VSS.RelSpeed:     Datapoint(rel_speed_acc),
            VSS.TTC:          Datapoint(ttc),
            VSS.HasTarget:    Datapoint(True),
            VSS.LeadSpeedEst: Datapoint(lead_speed_est),
        }
        kc.set_current_values(updates)   # ★ 센서는 current에 기록
        last_pub_ts = now
        # print(f"[RADAR→KUKSA] d={distance:.1f} rel={rel_speed_acc:+.1f} ttc={ttc:.1f} v_lead≈{lead_speed_est:.1f}")

    radar.listen(on_radar)

    # === 표시 루프 ===
    if args.display:
        cv2.namedWindow('front', cv2.WINDOW_NORMAL)
        if chase is not None:
            cv2.namedWindow('chase', cv2.WINDOW_NORMAL)

    print("[RUN] Streaming... (Ctrl+C to stop)")
    last_status = time.time()
    try:
        while True:
            world.tick()
            now = time.time()
            if now - last_status >= 5.0:
                v = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
                loc = ego.get_transform().location
                print(f"[STATUS] t={now:.0f}s ego@({loc.x:.1f},{loc.y:.1f}) {speed_kmh:.1f} km/h")
                last_status = now

            if args.display:
                if latest_front['bgr'] is not None:
                    cv2.imshow('front', latest_front['bgr'])
                if chase is not None and latest_chase['bgr'] is not None:
                    cv2.imshow('chase', latest_chase['bgr'])
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
        try: pub_cam.undeclare(); sess.close()
        except: pass
        try: kc.disconnect()
        except: pass
        world.apply_settings(original_settings)
        print("[CLEAN] Done.")

if __name__ == "__main__":
    main()
