#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, math, argparse
from typing import Optional
import numpy as np
import cv2
import carla
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint  # kuksa-client==0.4.0

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- 환경 변수 / 기본 파라미터 ---
# ==========================================================================
IMG_W        = int(os.environ.get("IMG_W", "640"))
IMG_H        = int(os.environ.get("IMG_H", "480"))
LEAD_GAP_M   = float(os.environ.get("LEAD_GAP_M", "30.0"))
STATUS_EVERY = float(os.environ.get("STATUS_EVERY", "5.0"))

ZENOH_ENDPOINT = os.environ.get("ZENOH_ENDPOINT", "tcp/127.0.0.1:7447")
KUKSA_ADDR     = os.environ.get("KUKSA_ADDR", "127.0.0.1:55555")
if ":" in KUKSA_ADDR:
    KUKSA_HOST, KUKSA_PORT = KUKSA_ADDR.split(":")[0], int(KUKSA_ADDR.split(":")[1])
else:
    KUKSA_HOST, KUKSA_PORT = KUKSA_ADDR, 55555
# ============================================================================


# --- 보조 함수 ---
# =====================================================================
def world_to_body(vx_w, vy_w, yaw_deg):
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return c*vx_w + s*vy_w, -s*vx_w + c*vy_w

def round2(x, n=6):
    return float(f"{x:.{n}f}")

def spawn_vehicle_round_robin(world, bp, start_idx=0, tries=60):
    """spawn_idx부터 여러 포인트를 순회하며 첫 성공을 반환"""
    sps = world.get_map().get_spawn_points()
    if not sps:
        raise RuntimeError("No spawn points on this map.")
    n = len(sps)
    idx0 = max(0, min(start_idx, n-1))
    last_err = None
    for k in range(tries):
        idx = (idx0 + k) % n
        tf  = sps[idx]
        try:
            actor = world.try_spawn_actor(bp, tf)
            if actor:
                return actor, idx
            last_err = RuntimeError(f"occupied at idx={idx}")
        except Exception as e:
            last_err = e
        # 한 틱 진행해 충돌/점유 해소
        try: world.tick()
        except Exception: time.sleep(0.02)
    raise RuntimeError(f"Failed to spawn after {tries} tries (start={idx0}). last={last_err}")

def attach_tm_autopilot(client, world, vehicle, ports=range(8000, 8012), sync=True):
    tm = None; tm_port = None
    for p in ports:
        try:
            tm = client.get_trafficmanager(p)
            if sync:
                tm.set_synchronous_mode(True)
            tm_port = p
            break
        except RuntimeError:
            tm = None
    if tm is not None:
        tm.auto_lane_change(vehicle, False)
        vehicle.set_autopilot(True, tm_port)
        print(f"[INFO] Lead autopilot ON via TM:{tm_port}")
    else:
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(throttle=0.20))
        print("[WARN] No TM port free, lead moves with constant low throttle.")
    return tm, tm_port

class DynPusher:
    """KUKSA 0.4.0 Datapoint 기반 동역학 퍼블리셔"""
    def __init__(self, host, port, enable=True):
        self.enable = enable
        self.prev = {"vx": None, "vy": None, "ts": None}
        self.cli: Optional[VSSClient] = None
        if enable:
            self.cli = VSSClient(host, port)
            self.cli.connect()
            print(f"[KUKSA] Connected to {host}:{port}")

    def close(self):
        if self.cli:
            try:
                self.cli.disconnect()
            finally:
                print("[KUKSA] Disconnected")

    def push(self, vehicle, sim_ts):
        if not self.cli:
            return
        v_w = vehicle.get_velocity()
        vx_w, vy_w = float(v_w.x), float(v_w.y)
        yaw_deg = float(vehicle.get_transform().rotation.yaw)
        vx_b, vy_b = world_to_body(vx_w, vy_w, yaw_deg)
        v_abs = math.hypot(vx_b, vy_b)

        ang = vehicle.get_angular_velocity()
        yaw_rate = float(ang.z)  # rad/s

        ax, ay = 0.0, 0.0
        if self.prev["ts"] is not None:
            dt = max(1e-6, sim_ts - self.prev["ts"])
            ax = (vx_b - (self.prev["vx"] or 0.0)) / dt
            ay = (vy_b - (self.prev["vy"] or 0.0)) / dt
        self.prev.update({"vx": vx_b, "vy": vy_b, "ts": sim_ts})

        odo_v_mean = v_abs

        updates = {
    "Vehicle.Private.Slip.Ts":                     Datapoint(float(sim_ts)),
    "Vehicle.Private.Slip.Metrics.v":              Datapoint(float(v_abs)),
    "Vehicle.Private.Slip.Metrics.vx":             Datapoint(float(vx_b)),
    "Vehicle.Private.Slip.Metrics.vy":             Datapoint(float(vy_b)),
    "Vehicle.Private.Slip.Metrics.ax_mean":        Datapoint(float(ax)),
    "Vehicle.Private.Slip.Metrics.ay_abs_mean":    Datapoint(float(ay)),
    "Vehicle.Private.Slip.Metrics.long_residual":  Datapoint(float(yaw_rate)),   # 임시 매핑
    "Vehicle.Private.Slip.Metrics.wheel_odo_v_mean": Datapoint(float(odo_v_mean)),

}

        self.cli.set_current_values(updates)

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--fps',  type=int, default=20)
    ap.add_argument('--spawn_idx', type=int, default=20)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=1)
    ap.add_argument('--record', type=str, default='')
    ap.add_argument('--record_mode', choices=['raw','vis','both'], default='vis')
    ap.add_argument('--no-kuksa', action='store_true')
    ap.add_argument('--no-lead', action='store_true')
    args = ap.parse_args()

    # ---------------- Zenoh 연결 (env.py 패턴) ----------------
    cfg = zenoh.Config()
    try:
        cfg.insert_json5('mode', '"client"')
        cfg.insert_json5('connect/endpoints', f'["{ZENOH_ENDPOINT}"]')
    except AttributeError:
        cfg.insert_json('mode', '"client"')
        cfg.insert_json('connect/endpoints', f'["{ZENOH_ENDPOINT}"]')
    sess = zenoh.open(cfg)
    pub  = sess.declare_publisher('carla/cam/front')

    print(f"[WORLD] preparing...")

    # ---------------- CARLA 연결 ----------------
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    fixed_dt = 1.0 / max(1, args.fps)
    original_settings = world.get_settings()

    # 월드 동기화 (env.py와 동일)
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds= fixed_dt
    settings.substepping        = False
    world.apply_settings(settings)
    print(f"[WORLD] synchronous_mode=True, fixed_delta_seconds={fixed_dt:.3f}")

    # ---------------- 차량 스폰 (안정화) ----------------
    bp = world.get_blueprint_library()

    # ego
    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', 'ego')
    sps = world.get_map().get_spawn_points()
    tf  = sps[min(max(0, args.spawn_idx), len(sps)-1)] if sps else None

    ego = world.try_spawn_actor(ego_bp, tf) if tf is not None else None
    if ego is None:
        # 점유/실패 시 라운드로빈으로 재시도
        try:
            ego, used_idx = spawn_vehicle_round_robin(world, ego_bp, start_idx=args.spawn_idx, tries=60)
            tf = sps[used_idx]
            print(f"[SPAWN] Ego via round-robin at idx={used_idx}")
        except Exception as e:
            world.apply_settings(original_settings)
            print(f"[ERR] Failed to spawn Ego. Try another spawn_idx or free the spawn point. ({e})")
            # Zenoh 정리
            try: pub.undeclare(); sess.close()
            except Exception: pass
            sys.exit(1)

    # lead + TM (env.py 스타일)
    lead = None; tm = None; tm_port = None
    if not args.no_lead:
        lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
        lead_bp.set_attribute('role_name', 'lead')
        ego_wp = world.get_map().get_waypoint(tf.location)
        lead_wp = ego_wp.next(LEAD_GAP_M)[0]
        lead_tf = lead_wp.transform
        lead_tf.location.z = tf.location.z
        lead = world.try_spawn_actor(lead_bp, lead_tf)
        if lead:
            tm, tm_port = attach_tm_autopilot(client, world, lead, sync=True)

    # ---------------- 센서 부착 (env.py 패턴) ----------------
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
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            latest_chase['bgr'] = arr[:, :, :3].copy()

        chase.listen(_on_chase)
        print("[SENSOR] Chase camera attached.")
    except Exception as e:
        print(f"[WARN] Failed to attach chase camera: {e}")

    # front camera
    latest_front = {'bgr': None}
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(fixed_dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # radar
    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120')
    radar_bp.set_attribute('horizontal_fov', '20')
    radar_bp.set_attribute('vertical_fov', '10')
    radar_bp.set_attribute('sensor_tick', str(fixed_dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar    = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)

    # ---------------- Zenoh 퍼블리시 콜백 ----------------
    def on_cam(img: carla.Image):
        # BGRA 원 버퍼를 memoryview로 잡고, bytes로 1회 변환(파이썬 특성상 최소 1회 복사)
        buf = memoryview(img.raw_data)
        att = json.dumps({
            "w": img.width,
            "h": img.height,
            "c": 4,
            "format": "bgra8",
            "stride": img.width * 4,
            "frame": int(img.frame),
            "sim_ts": float(img.timestamp),   # 심시간
            "pub_ts": time.time()             # 벽시계
        }).encode("utf-8")
        pub.put(bytes(buf), attachment=att)

        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest_front['bgr'] = arr[:, :, :3].copy()

        print(f"[CAM] frame={img.frame:06d} sim_ts={img.timestamp:.3f} bytes={len(buf)}")

    cam.listen(on_cam)

    def on_radar(meas: carla.RadarMeasurement):
        count = len(meas)
        nearest = min((d.depth for d in meas), default=None)
        if nearest is not None:
            print(f"[RADAR] frame={meas.frame:06d} detections={count} nearest_depth={nearest:.2f}m")
        else:
            print(f"[RADAR] frame={meas.frame:06d} detections={count}")

    radar.listen(on_radar)

    # ---------------- 창 생성 (env.py 스타일) ----------------
    if args.display:
        cv2.namedWindow('front', cv2.WINDOW_NORMAL)
        cv2.namedWindow('chase', cv2.WINDOW_NORMAL)

    # ---------------- KUKSA 퍼블리셔 ----------------
    dyn = DynPusher(KUKSA_HOST, KUKSA_PORT, enable=(not args.no_kuksa))

    print("[RUN] Streaming... (Ctrl+C to stop)")
    try:
        last_status = time.time()
        while True:
            world.tick()
            ts = float(world.get_snapshot().timestamp.elapsed_seconds)

            # 동역학 퍼블리시 (심시간 ts)
            try:
                dyn.push(ego, ts)
            except Exception as e:
                print(f"[KUKSA] push error: {e}")

            now = time.time()
            if now - last_status >= STATUS_EVERY:
                v = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
                loc = ego.get_transform().location
                print(f"[STATUS] t={int(now)}s ego@({loc.x:.1f},{loc.y:.1f}) {speed_kmh:.1f} km/h")
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
        # 센서 stop
        try:
            cam.stop(); radar.stop()
            if chase is not None: chase.stop()
        except Exception:
            pass
        # 액터 destroy
        for a in [cam, radar, chase, lead, ego]:
            if a is not None:
                try: a.destroy()
                except Exception: pass
        # KUKSA/Zenoh/월드 복구
        try: dyn.close()
        except Exception: pass
        try:
            pub.undeclare()
            sess.close()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        if args.display:
            try: cv2.destroyAllWindows()
            except Exception: pass
        print("[CLEAN] Done.")

if __name__ == "__main__":
    main()

