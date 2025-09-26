#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
control.py — Passive controller (Env가 world.tick 담당)
- Ego: Autopilot OFF, Kuksa에서 ACC/LK 명령 'current'를 읽어 CARLA에 적용
- Lead: TrafficManager Autopilot ON (옵션)
- Kuksa 경로:
    * Throttle/Brake/Mode: Vehicle.ADAS.ACC.Ctrl.*
    * Steer: Vehicle.ADAS.ACC.Ctrl.Steer(있으면 우선) → 없으면 Vehicle.ADAS.LK.Steering
"""

import time
import math
import argparse
import traceback
from typing import Any, Optional
from queue import Queue
import carla

# --- Kuksa (grpc) ---
DatapointType = None
try:
    from kuksa_client.grpc import VSSClient as DataBrokerClient
    try:
        from kuksa_client.grpc import Datapoint as _DP
        DatapointType = _DP
    except Exception:
        DatapointType = None
except Exception:
    DataBrokerClient = None
    DatapointType = None


# -------------------- 유틸 --------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def to_float_safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        if DatapointType is not None and isinstance(x, DatapointType):
            try: return float(getattr(x, "value", default))
            except Exception: return default
        if isinstance(x, dict) and "value" in x:
            try: return float(x["value"])
            except Exception: return default
        if isinstance(x, (list, tuple)) and x:
            return to_float_safe(x[0], default)
        return default

def unwrap(x: Any):
    if DatapointType is not None and isinstance(x, DatapointType):
        return getattr(x, "value", None)
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    if isinstance(x, (list, tuple)) and x:
        return unwrap(x[0])
    return x

def speed_kmh(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def find_actor_by_role(world: carla.World, role: str, wait_sec: float = 15.0) -> Optional[carla.Actor]:
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        for v in world.get_actors().filter("*vehicle*"):
            if v.attributes.get("role_name") == role:
                return v
        time.sleep(0.05)
    return None


# -------------------- Kuksa 래퍼 --------------------
class KUKSA:
    def __init__(self, host="127.0.0.1", port=55555):
        self.cli = None
        if DataBrokerClient is None:
            print("[KUKSA] sdk not available → running with zeros")
            return
        try:
            try:
                self.cli = DataBrokerClient(host, port)
            except Exception:
                self.cli = DataBrokerClient(f"{host}:{port}")
            if hasattr(self.cli, "connect"):
                self.cli.connect()
            print(f"[KUKSA] connected @ {host}:{port}")
        except Exception as e:
            print("[KUKSA] connect failed:", e)
            self.cli = None

    def get_current_map(self, keys):
        if self.cli is None: return {}
        try:
            res = self.cli.get_current_values(keys)
            return res if isinstance(res, dict) else {}
        except Exception:
            return {k: None for k in keys}


# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser("control (ACC+LK via Kuksa, current-only)")
    # CARLA
    ap.add_argument("--carla_host", default="127.0.0.1")
    ap.add_argument("--carla_port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)
    ap.add_argument("--ego_role", default="ego")
    ap.add_argument("--lead_role", default="lead")
    # Kuksa
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)
    # VSS keys
    ap.add_argument("--thr_key",  default="Vehicle.ADAS.ACC.Ctrl.Throttle")
    ap.add_argument("--brk_key",  default="Vehicle.ADAS.ACC.Ctrl.Brake")
    ap.add_argument("--mode_key", default="Vehicle.ADAS.ACC.Ctrl.Mode")
    ap.add_argument("--steer_acc_key", default="Vehicle.ADAS.ACC.Ctrl.Steer")
    ap.add_argument("--steer_lk_key",  default="Vehicle.ADAS.LK.Steering")
    ap.add_argument("--prefer_acc_steer", type=int, default=0, help="1이면 ACC.Ctrl.Steer 우선")
    # Filters
    ap.add_argument("--steer_deadband", type=float, default=0.04)
    ap.add_argument("--steer_alpha", type=float, default=0.25, help="EMA 계수(0이면 미사용)")
    ap.add_argument("--steer_rate", type=float, default=2.0, help="레이트리밋(노멀라이즈/초, 0=off)")
    ap.add_argument("--thr_alpha", type=float, default=0.2, help="스로틀 EMA(0=off)")
    ap.add_argument("--brk_alpha", type=float, default=0.2, help="브레이크 EMA(0=off)")
    # 로그
    ap.add_argument("--log_every", type=float, default=0.5)
    args = ap.parse_args()

    # CARLA 접속 (Env가 sync 설정/틱 담당)
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world: carla.World = client.get_world()
    settings = world.get_settings()
    print(f"[WORLD] connected. sync={settings.synchronous_mode}, fixed_dt={settings.fixed_delta_seconds}")

    ego = find_actor_by_role(world, args.ego_role)
    if not ego:
        raise RuntimeError(f"Ego(role='{args.ego_role}') not found")
    lead = find_actor_by_role(world, args.lead_role)

    # 역할 설정
    try: ego.set_autopilot(False)
    except Exception: pass
    if lead is not None:
        try:
            tm = client.get_trafficmanager(args.tm_port)
            try: tm.set_synchronous_mode(True)
            except Exception: pass
            lead.set_autopilot(True, args.tm_port)
            try:
                if hasattr(tm, "auto_lane_change"): tm.auto_lane_change(lead, False)
            except Exception: pass
            print("[INFO] Lead autopilot ON via TM")
        except Exception as e:
            print("[TM] lead setup failed:", e)

    kuksa = KUKSA(args.kuksa_host, args.kuksa_port)

    # 패시브 페이싱(Env tick 기반)
    tick_q: Queue = Queue(maxsize=3)
    world.on_tick(tick_q.put)

    # 필터 상태
    steer_ema = None if args.steer_alpha <= 0.0 else 0.0
    thr_ema = None if args.thr_alpha   <= 0.0 else 0.0
    brk_ema = None if args.brk_alpha   <= 0.0 else 0.0
    steer_prev = 0.0
    last_log = time.time()

    print(f"[RUN] Control via Kuksa (current-only) | steer={'ACC.Ctrl.Steer' if args.prefer_acc_steer else 'LK.Steering'}→fallback")
    try:
        while True:
            snap: carla.Timestamp = tick_q.get()
            dt = snap.delta_seconds if snap and snap.delta_seconds else (settings.fixed_delta_seconds or 0.05)

            # --- Kuksa 현재값 일괄 읽기 ---
            keys = [args.thr_key, args.brk_key, args.mode_key, args.steer_acc_key, args.steer_lk_key]
            vals = kuksa.get_current_map(keys)

            def _val(k, default=0.0):
                return to_float_safe(unwrap(vals.get(k)), default)

            thr_raw = _val(args.thr_key, 0.0)
            brk_raw = _val(args.brk_key, 0.0)
            mode    = unwrap(vals.get(args.mode_key))

            # steer: ACC 우선 옵션
            steer_raw = None
            if args.prefer_acc_steer:
                sr = vals.get(args.steer_acc_key)
                if sr is not None:
                    steer_raw = _val(args.steer_acc_key, 0.0)
            if steer_raw is None:
                steer_raw = _val(args.steer_lk_key, 0.0)

            # --- 클램프/필터 ---
            thr = clamp(thr_raw, 0.0, 1.0)
            brk = clamp(brk_raw, 0.0, 1.0)

            if thr_ema is not None:
                thr_ema = args.thr_alpha * thr + (1.0 - args.thr_alpha) * thr_ema
                thr = clamp(thr_ema, 0.0, 1.0)
            if brk_ema is not None:
                brk_ema = args.brk_alpha * brk + (1.0 - args.brk_alpha) * brk_ema
                brk = clamp(brk_ema, 0.0, 1.0)

            steer = clamp(steer_raw, -1.0, 1.0)
            if abs(steer) < args.steer_deadband:
                steer = 0.0
            if steer_ema is not None:
                steer_ema = args.steer_alpha * steer + (1.0 - args.steer_alpha) * steer_ema
                steer = clamp(steer_ema, -1.0, 1.0)
            if args.steer_rate > 0.0 and dt and dt > 0.0:
                max_step = args.steer_rate * dt
                steer = clamp(steer, steer_prev - max_step, steer_prev + max_step)
            steer_prev = steer

            # 스로틀/브레이크 충돌 방지(필요 시)
            if brk > 0.01 and thr > 0.01:
                thr = 0.0  # 보수적

            # --- CARLA 적용 ---
            ego.apply_control(carla.VehicleControl(
                throttle=float(thr),
                brake=float(brk),
                steer=float(steer),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            ))

            # --- 로그 ---
            now = time.time()
            if now - last_log >= max(0.1, args.log_every):
                v = speed_kmh(ego)
                m = str(mode) if mode is not None else "UNKNOWN"
                print(f"[CTL] v={v:5.1f} km/h | thr={thr:4.2f} brk={brk:4.2f} steer={steer:+.3f} | mode={m}")
                last_log = now

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    except Exception:
        traceback.print_exc()
    finally:
        try: ego.set_autopilot(False)
        except Exception: pass
        print("[CLEAN] control.py passive exit")


if __name__ == "__main__":
    main()
