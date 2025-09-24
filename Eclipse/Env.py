import os, sys, time, json, math, argparse
from typing import Optional

import numpy as np
import cv2
import carla
import zenoh
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- Zenoh 환경 구축 ---
# =======================================================
cfg = zenoh.Config()
try:
    cfg.insert_json5('mode', '"client"')
    cfg.insert_json5('connect/endpoints', '["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json('mode', '"client"')
    cfg.insert_json('connect/endpoints', '["tcp/127.0.0.1:7447"]')

sess = zenoh.open(cfg)
pub  = sess.declare_publisher('carla/cam/front')
# ========================================================


# --- 기본 파라미터 ---
# =======================================================
IMG_W        = int(os.environ.get("IMG_W", "640"))
IMG_H        = int(os.environ.get("IMG_H", "480"))
SENSOR_TICK  = float(os.environ.get("SENSOR_TICK", "0.05"))   # 20Hz
LEAD_GAP_M   = float(os.environ.get("LEAD_GAP_M", "30.0"))
STATUS_EVERY = float(os.environ.get("STATUS_EVERY", "5.0"))
# ========================================================


# --- main 문 ---
# ========================================================
def main():
    # ------------------- Argparse 인자 파싱 ----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--fps',  type=int, default=20)
    ap.add_argument('--spawn_idx', type=int, default=328)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=1)
    ap.add_argument('--record', type=str, default='')
    ap.add_argument('--record_mode', choices=['raw','vis','both'], default='vis')
    args = ap.parse_args()

    # ---------------- CARLA 연결 ----------------
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    dt = 1.0 / max(1, args.fps)
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.substepping = False
    world.apply_settings(settings)
    print(f"[WORLD] synchronous_mode=True, delta_seconds={dt:.3f}")

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

    # lead (TM 없이 스폰만 하고 정지 유지)
    lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
    lead_bp.set_attribute('role_name', 'lead')
    ego_wp = world.get_map().get_waypoint(tf.location)
    lead_wp = ego_wp.next(30.0)[0]
    lead_tf = lead_wp.transform
    lead_tf.location.z = tf.location.z
    lead = world.try_spawn_actor(lead_bp, lead_tf)
    if lead is None:
        raise RuntimeError("Failed to spawn Lead. Try another spawn_idx or free the spawn point.")

    # ---------------- 센서 부착 ----------------
    # chase camera(레이더용 버드 뷰)
    chase = None
    latest_chase = {'bgr': None}
    try:
        chase_bp = bp.find('sensor.camera.rgb')
        chase_bp.set_attribute('image_size_x', str(args.width))
        chase_bp.set_attribute('image_size_y', str(args.height))
        chase_bp.set_attribute('fov', '70')
        chase_bp.set_attribute('sensor_tick', str(dt))
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

    # 카메라 센서
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120')
    radar_bp.set_attribute('horizontal_fov', '20')
    radar_bp.set_attribute('vertical_fov', '10')
    radar_bp.set_attribute('sensor_tick', str(dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar    = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)

    # ---------------- 카메라 → Zenoh 전송 ----------------
    def on_cam(img: carla.Image):       # 들어온 이미지를 BGR 배열로 변환
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

        print(f"[CAM] image sending...")

    cam.listen(on_cam)

    # ---------------- 레이더 로그 ----------------
    ''' 
    <기진 수정 부탁>
    - 현재는 가장 가까운 레이더 값만 수진하게 해놓음
    - 노이즈 제거 및 고정물체 필터 같은 로직을 여기에 구현하면 좋을 듯?
    '''
    def on_radar(meas: carla.RadarMeasurement):
        if len(meas) == 0:
            # 타겟 없음 (KUKSA 제거: 상태 publish 생략)
            return

        # 가장 가까운 detection만 선택 (예: 최소 depth)
        best = min(meas, key=lambda d: d.depth)

        distance = float(best.depth)      # m
        rel_speed = float(best.velocity)  # m/s (CARLA convention: +면 멀어짐, -면 접근)

        # KUKSA 제거: publish 생략, 로그만 유지
        print(f"[RADAR] dist={distance:.1f}/ m rel_speed={rel_speed:+.1f}m/s")

    radar.listen(on_radar)

    # 창 생성 
    if args.display:
        cv2.namedWindow('front', cv2.WINDOW_NORMAL)
        cv2.namedWindow('chase', cv2.WINDOW_NORMAL)

    latest_front = {'bgr': None}
    print("[RUN] Streaming... (Ctrl+C to stop)")

    # ---------------------- 메인 로직 루프 -----------------------------
    try:
        last_status = time.time()
        while True:
            world.tick()
            now = time.time()
            if now - last_status >= STATUS_EVERY:       # 5초마다 주기적 상태 로그 프린트
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

    # ----------- 종료 로직 ------------
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
