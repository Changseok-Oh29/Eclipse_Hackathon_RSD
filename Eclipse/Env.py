import os, sys, time, json, math, argparse
from typing import Optional
import numpy as np
import cv2
import carla
import zenoh
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

cfg = zenoh.Config()
try:
    cfg.insert_json5('mode', '"client"')
    cfg.insert_json5('connect/endpoints', '["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json('mode', '"client"')
    cfg.insert_json('connect/endpoints', '["tcp/127.0.0.1:7447"]')

sess = zenoh.open(cfg)
pub  = sess.declare_publisher('carla/cam/front')

# =========================
# 기본 파라미터
# =========================
IMG_W        = int(os.environ.get("IMG_W", "640"))
IMG_H        = int(os.environ.get("IMG_H", "480"))
SENSOR_TICK  = float(os.environ.get("SENSOR_TICK", "0.05"))   # 20Hz
LEAD_GAP_M   = float(os.environ.get("LEAD_GAP_M", "30.0"))
STATUS_EVERY = float(os.environ.get("STATUS_EVERY", "5.0"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--fps',  type=int, default=20)
    ap.add_argument('--spawn_idx', type=int, default=20)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=580)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=1)
    ap.add_argument('--record', type=str, default='')
    ap.add_argument('--record_mode', choices=['raw','vis','both'], default='vis')
    args = ap.parse_args()

    # ---------------- CARLA 연결 ----------------
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    fixed_dt = 1.0 / max(1, args.fps)
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    settings.substepping = False
    world.apply_settings(settings)
    print(f"[WORLD] synchronous_mode=True, fixed_delta_seconds={fixed_dt:.3f}")

    # ---------------- 차량 스폰 ----------------
    bp = world.get_blueprint_library()
    # ego
    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', 'ego')
    sps = world.get_map().get_spawn_points()
    tf  = sps[min(max(0, args.spawn_idx), len(sps)-1)]
    ego = world.try_spawn_actor(ego_bp, tf)
    if ego is None:
        raise RuntimeError("Failed to spawn Ego. Try another spawn_idx or free the spawn point.")

    # lead + TM
    lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
    lead_bp.set_attribute('role_name', 'lead')
    ego_wp = world.get_map().get_waypoint(tf.location)
    lead_wp = ego_wp.next(30.0)[0]
    lead_tf = lead_wp.transform
    lead_tf.location.z = tf.location.z
    lead = world.try_spawn_actor(lead_bp, lead_tf)
    if lead:
        for port in range(8000, 8010):
            try:
                tm = client.get_trafficmanager(port); tm.set_synchronous_mode(True); tm_port = port; break
            except RuntimeError:
                tm = None
        if tm is not None:
            tm.auto_lane_change(lead, False)
            lead.set_autopilot(True, tm_port)
            print(f"[INFO] Lead autopilot ON via TM:{tm_port}")
        else:
            lead.set_autopilot(False)
            lead.apply_control(carla.VehicleControl(throttle=0.20))
            print("[WARN] No TM port free, lead moves with constant low throttle.")

    # chase camera
    chase = None
    latest_chase = {'bgr': None}
    try:
        chase_bp = bp.find('sensor.camera.rgb')
        chase_bp.set_attribute('image_size_x', str(args.width))
        chase_bp.set_attribute('image_size_y', str(args.height))
        chase_bp.set_attribute('fov', '70')
        chase_bp.set_attribute('sensor_tick', str(fixed_dt))
        chase_tf = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12.0))
        chase = world.spawn_actor(chase_bp, chase_tf, attach_to=ego)

        def _on_chase(img: carla.Image):
            # BGRA → BGR
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            latest_chase['bgr'] = arr[:, :, :3].copy()

        chase.listen(_on_chase)
        print("[SENSOR] Chase camera attached.")
    except Exception as e:
        print(f"[WARN] Failed to attach chase camera: {e}")

    # ---------------- 센서 부착 ----------------
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(fixed_dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120')
    radar_bp.set_attribute('horizontal_fov', '20')
    radar_bp.set_attribute('vertical_fov', '10')
    radar_bp.set_attribute('sensor_tick', str(fixed_dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar    = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)

    # ---------------- 콜백: 카메라 → Zenoh 전송 ----------------
    def on_cam(img: carla.Image):
        # BGRA 원 버퍼를 memoryview로 잡고, bytes로 1회 변환(파이썬 특성상 최소 1회 복사)
        buf = memoryview(img.raw_data)
        # 메타데이터(수신측 해석용)
        att = json.dumps({
            "w": img.width,
            "h": img.height,
            "c": 4,
            "format": "bgra8",
            "stride": img.width * 4,
            "frame": int(img.frame),
            "sim_ts": float(img.timestamp),
            "pub_ts": time.time()
        }).encode("utf-8")
        pub.put(bytes(buf), attachment=att)

        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest_front['bgr'] = arr[:, :, :3].copy()

        print(f"[CAM] frame={img.frame:06d} sim_ts={img.timestamp:.3f} bytes={len(buf)}")

    cam.listen(on_cam)

    # ---------------- 콜백: 레이더 로그 ----------------
    def on_radar(meas: carla.RadarMeasurement):
        count = len(meas)
        nearest = min((d.depth for d in meas), default=None)
        if nearest is not None:
            print(f"[RADAR] frame={meas.frame:06d} detections={count} nearest_depth={nearest:.2f}m")
        else:
            print(f"[RADAR] frame={meas.frame:06d} detections={count}")

    radar.listen(on_radar)

    # 창 생성 
    if args.display:
        cv2.namedWindow('front', cv2.WINDOW_NORMAL)
        cv2.namedWindow('chase', cv2.WINDOW_NORMAL)

    latest_front = {'bgr': None}
    print("[RUN] Streaming... (Ctrl+C to stop)")
    try:
        last_status = time.time()
        while True:
            world.tick()
            now = time.time()
            if now - last_status >= STATUS_EVERY:
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

            time.sleep(0.005)  # CPU 여유
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received, cleaning up...")
    finally:
        try:
            cam.stop()
            radar.stop()
        except Exception:
            pass
        for a in [cam, radar, lead, ego]:
            if a is not None:
                try:
                    a.destroy()
                except Exception:
                    pass
        try:
            pub.undeclare()
            sess.close()
        except Exception:
            pass
        world.apply_settings(original_settings) 
        print("[CLEAN] Done.")

if __name__ == "__main__":
    main()
