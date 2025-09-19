#!/usr/bin/env python3
# decision1.py — LKAS + ACC (ACC 분리 + ego 대기 + GO ping)

import argparse, json, time
from queue import Queue, Empty
import numpy as np
import carla
import zenoh

from common.LK_algo import gains_for_speed, speed_of
from common.acc_algo import ACCController, ACCInputs


# ------------------------
# 유틸 함수
# ------------------------
def find_by_role(world, role='ego'):
    actors = world.get_actors().filter('vehicle.*')
    for v in actors:
        if v.attributes.get('role_name') == role:
            return v
    return None

def find_vehicle_from_camera_parent(world):
    sensors = world.get_actors().filter('sensor.*')
    for s in sensors:
        parent = s.parent
        if parent and parent.type_id.startswith("vehicle."):
            return parent
    return None

def wait_for_ego(world, role: str, timeout_sec: float = 20.0):
    """role 이름 차량 or 카메라 부모 차량을 timeout까지 polling해서 반환."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        v = find_by_role(world, role)
        if v is not None:
            return v
        v = find_vehicle_from_camera_parent(world)
        if v is not None:
            return v
        time.sleep(0.25)
    return None


# ------------------------
# 메인
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--role', default='ego')
    ap.add_argument('--ego_timeout', type=float, default=20.0, help='ego spawn wait seconds')
    ap.add_argument('--deadband', type=int, default=4)
    ap.add_argument('--apply_hz', type=float, default=20.0)
    args = ap.parse_args()

    # CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    # ego 대기 (perception이 늦게 스폰해도 OK)
    ego = wait_for_ego(world, args.role, timeout_sec=args.ego_timeout)
    if ego is None:
        raise RuntimeError(f"Ego vehicle not found (role='{args.role}')!")

    try:
        ego.set_autopilot(False)
    except Exception:
        pass

    # ------------------------
    # Zenoh pub/sub 설정
    # ------------------------
    z = zenoh.open({})
    q_lk = Queue(maxsize=200)
    def _on_lk(s):
        try:
            q_lk.put_nowait(s)
        except:
            pass
    sub_lk = z.declare_subscriber('demo/lk/features', _on_lk)

    q_acc = Queue(maxsize=200)
    def _on_acc(s):
        try:
            q_acc.put_nowait(s)
        except:
            pass
    sub_acc = z.declare_subscriber('demo/acc/features', _on_acc)

    pub_ctrl = z.declare_publisher('demo/lk/ctrl')

    # perception이 wait_for_decision=1일 때 출발하도록 GO ping 1회
    pub_ctrl.put(json.dumps({
        "throttle": 0.0, "brake": 0.0, "steer": 0.0,
        "mode": "CRUISE", "ts": time.time()
    }).encode('utf-8'))

    # ------------------------
    # 상태 변수
    # ------------------------
    last_w = None
    last_x_mid = None

    acc_dist = None
    acc_rel_v = 0.0
    acc_ttc = float('inf')
    acc_lead_speed_est = None
    acc_has_target = False

    current_mode = "CRUISE"
    period = 1.0 / max(1e-3, args.apply_hz)
    next_t = time.perf_counter() + period
    log_next = time.perf_counter() + 1.0
    sim_time = 0.0

    # ACC 컨트롤러
    acc_ctrl = ACCController(apply_hz=args.apply_hz)

    print(f"[INFO] Decision controlling id={ego.id}, role={ego.attributes.get('role_name','')}")
    print("[INFO] Decision running... Ctrl+C to stop.")

    try:
        while True:
            # 최신 피처 수신 (LKAS)
            try:
                while True:
                    s = q_lk.get_nowait()
                    feat = json.loads(bytes(s.payload).decode('utf-8'))
                    last_w = feat.get('w', last_w)
                    last_x_mid = feat.get('x_lane_mid', last_x_mid)
            except Empty:
                pass

            # 최신 피처 수신 (ACC)
            try:
                while True:
                    s = q_acc.get_nowait()
                    feat = json.loads(bytes(s.payload).decode('utf-8'))
                    d   = feat.get('distance', None)
                    acc_dist = float(d) if d is not None else None
                    acc_rel_v = float(feat.get('rel_speed', acc_rel_v))
                    ttc = feat.get('ttc', None)
                    acc_ttc = float(ttc) if ttc is not None else float('inf')
                    ls = feat.get('lead_speed_est', None)
                    acc_lead_speed_est = float(ls) if ls is not None else None
                    acc_has_target = bool(feat.get('has_target', False))
            except Empty:
                pass

            # 페이싱
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period
            sim_time += period

            # LKAS 제어
            v_mps = speed_of(ego)
            kp, clip = gains_for_speed(v_mps)
            steer = 0.0
            if (last_w is not None) and (last_x_mid is not None):
                err_px = (last_w / 2.0) - float(last_x_mid)
                if abs(err_px) < args.deadband:
                    err_px = 0.0
                steer = float(np.clip(kp * -err_px, -clip, clip))

            # ACC 제어 (분리 모듈)
            acc_in = ACCInputs(
                distance=acc_dist,
                rel_speed=acc_rel_v,
                ttc=(acc_ttc if np.isfinite(acc_ttc) else None),
                has_target=acc_has_target,
                lead_speed_est=acc_lead_speed_est
            )
            throttle_cmd, brake_cmd, current_mode, dbg = acc_ctrl.step(sim_time, v_mps, acc_in)

            # 차량 제어 적용
            ego.apply_control(carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=float(np.clip(steer, -1.0, 1.0)),
                hand_brake=False, manual_gear_shift=False
            ))

            # 컨트롤 퍼블리시
            ctrl_msg = {
                "throttle": float(throttle_cmd),
                "brake": float(brake_cmd),
                "steer": float(steer),
                "mode": current_mode,
                "ts": time.time()
            }
            pub_ctrl.put(json.dumps(ctrl_msg).encode('utf-8'))

            # 디버그 로그
            if now >= log_next:
                distance = dbg["distance"]; rel = dbg["rel_v"]; ttc = dbg["ttc"]
                print(f"[DBG] mode={current_mode:6s}  v={v_mps*3.6:5.1f} km/h  "
                      f"tgt_ready={int(dbg['target_ready'])}  dist={(distance if np.isfinite(distance) else -1):5.1f}  "
                      f"rel={float(rel):+4.1f}  ttc={(ttc if np.isfinite(ttc) else -1):4.1f}  "
                      f"thr={throttle_cmd:.2f}  brk={brake_cmd:.2f}  steer={steer:+.3f}")
                log_next = now + 1.0

    finally:
        # 정리
        try: sub_lk.undeclare()
        except: pass
        try: sub_acc.undeclare()
        except: pass
        try: pub_ctrl.undeclare()
        except: pass
        try: z.close()
        except: pass


# ------------------------
# 엔트리포인트
# ------------------------
if __name__ == "__main__":
    try:
        print("[BOOT] decision1 starting...")
        main()
    except KeyboardInterrupt:
        print("[EXIT] KeyboardInterrupt")
    except Exception as e:
        import traceback, sys
        print("[FATAL] Uncaught exception:", e, file=sys.stderr)
        traceback.print_exc()

