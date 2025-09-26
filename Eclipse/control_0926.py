#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
control_acc_kuksa.py — Passive controller
- Ego: Autopilot OFF, Kuksa 명령(ACC/LK) 읽어 CARLA에 적용
- Lead: TM Autopilot ON (옵션, 절대속도 지정 가능)
- Kuksa 읽기는 target 우선, 없으면 current 폴백
"""

import time, math, argparse, traceback
from typing import Any, Optional
from queue import Queue
import carla

# ---- Kuksa ----
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

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def to_float_safe(x, default=0.0):
    try: return float(x)
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
def unwrap(x):
    if DatapointType is not None and isinstance(x, DatapointType):
        return getattr(x, "value", None)
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    if isinstance(x, (list, tuple)) and x:
        return unwrap(x[0])
    return x
def speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
def find_actor_by_role(world, role, wait_sec=15.0):
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        for v in world.get_actors().filter("*vehicle*"):
            if v.attributes.get("role_name") == role:
                return v
        time.sleep(0.05)
    return None

class KUKSA:
    def __init__(self, host="127.0.0.1", port=55555):
        self.cli = None
        if DataBrokerClient is None:
            print("[KUKSA] sdk not available → zeros"); return
        try:
            try: self.cli = DataBrokerClient(host, port)
            except Exception: self.cli = DataBrokerClient(f"{host}:{port}")
            if hasattr(self.cli, "connect"): self.cli.connect()
            print(f"[KUKSA] connected @ {host}:{port}")
        except Exception as e:
            print("[KUKSA] connect failed:", e); self.cli = None

    def _get_map(self, getter_name, keys):
        if self.cli is None: return {}
        try: getter = getattr(self.cli, getter_name)
        except Exception: return {}
        try:
            res = getter(keys)
            if isinstance(res, dict): return res
            return {k: getattr(self.cli, getter_name[:-1], lambda kk: None)(k) for k in keys}
        except Exception:
            return {}

    def read_value(self, key, prefer_target=True):
        if self.cli is None: return None
        if prefer_target and hasattr(self.cli, "get_target_values"):
            m = self._get_map("get_target_values", [key])
            if key in m: return unwrap(m[key])
        if hasattr(self.cli, "get_current_values"):
            m = self._get_map("get_current_values", [key])
            if key in m: return unwrap(m[key])
        for name in ("get_target_value","get_current_value","get_value","get","read"):
            if hasattr(self.cli, name):
                try: return unwrap(getattr(self.cli, name)(key))
                except Exception: pass
        return None

def main():
    ap = argparse.ArgumentParser("control (ACC+LK via Kuksa)")
    # CARLA
    ap.add_argument("--carla_host", default="127.0.0.1")
    ap.add_argument("--carla_port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)
    ap.add_argument("--ego_role", default="ego")
    ap.add_argument("--lead_role", default="lead")
    ap.add_argument("--lead_speed_kmh", type=float, default=15.0, help="TM 리드 절대 속도")
    # Kuksa
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)
    # VSS keys
    ap.add_argument("--thr_key",  default="Vehicle.ADAS.ACC.Ctrl.Throttle")
    ap.add_argument("--brk_key",  default="Vehicle.ADAS.ACC.Ctrl.Brake")
    ap.add_argument("--mode_key", default="Vehicle.ADAS.ACC.Ctrl.Mode")
    ap.add_argument("--steer_acc_key", default="Vehicle.ADAS.ACC.Ctrl.Steer")
    ap.add_argument("--steer_lk_key",  default="Vehicle.ADAS.LK.Steering")
    ap.add_argument("--prefer_acc_steer", type=int, default=1)
    # Filters
    ap.add_argument("--steer_deadband", type=float, default=0.03)
    ap.add_argument("--steer_alpha", type=float, default=0.40)
    ap.add_argument("--steer_rate", type=float, default=2.0)
    ap.add_argument("--thr_alpha", type=float, default=0.0)  # 반응 빠르게 기본 0
    ap.add_argument("--brk_alpha", type=float, default=0.0)
    ap.add_argument("--log_every", type=float, default=0.5)
    args = ap.parse_args()

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world = client.get_world()
    settings = world.get_settings()
    print(f"[WORLD] connected. sync={settings.synchronous_mode}, fixed_dt={settings.fixed_delta_seconds}")

    ego = find_actor_by_role(world, args.ego_role)
    if not ego: raise RuntimeError(f"Ego(role='{args.ego_role}') not found")
    lead = find_actor_by_role(world, args.lead_role)

    try: ego.set_autopilot(False)
    except Exception: pass
    if lead is not None:
        try:
            tm = client.get_trafficmanager(args.tm_port)
            try: tm.set_synchronous_mode(True)
            except Exception: pass
            lead.set_autopilot(True, args.tm_port)
            try: tm.auto_lane_change(lead, False)
            except Exception: pass
            try: tm.set_desired_speed(lead, float(args.lead_speed_kmh))
            except Exception: pass
            print("[INFO] Lead autopilot ON via TM")
        except Exception as e:
            print("[TM] lead setup failed:", e)

    kuksa = KUKSA(args.kuksa_host, args.kuksa_port)

    tick_q: Queue = Queue(maxsize=3)
    world.on_tick(tick_q.put)

    steer_ema = None if args.steer_alpha <= 0.0 else 0.0
    thr_ema = None if args.thr_alpha   <= 0.0 else 0.0
    brk_ema = None if args.brk_alpha   <= 0.0 else 0.0
    steer_prev = 0.0
    last_log = time.time()

    print(f"[RUN] Control via Kuksa (target→current fallback) | steer={'ACC.Ctrl.Steer' if args.prefer_acc_steer else 'LK.Steering'}→LK fallback")
    try:
        while True:
            snap = tick_q.get()
            dt = snap.delta_seconds if snap and snap.delta_seconds else (settings.fixed_delta_seconds or 0.05)

            thr_raw = to_float_safe(kuksa.read_value(args.thr_key,  prefer_target=True), 0.0)
            brk_raw = to_float_safe(kuksa.read_value(args.brk_key,  prefer_target=True), 0.0)
            _mode   = kuksa.read_value(args.mode_key,  prefer_target=True)

            steer_raw = None
            if args.prefer_acc_steer:
                v = kuksa.read_value(args.steer_acc_key, prefer_target=True)
                if v is not None: steer_raw = to_float_safe(v, 0.0)
            if steer_raw is None:
                v = kuksa.read_value(args.steer_lk_key, prefer_target=True)
                steer_raw = to_float_safe(v, 0.0)

            thr = clamp(thr_raw, 0.0, 1.0)
            brk = clamp(brk_raw, 0.0, 1.0)
            if thr_ema is not None:
                thr_ema = args.thr_alpha * thr + (1.0 - args.thr_alpha) * thr_ema
                thr = clamp(thr_ema, 0.0, 1.0)
            if brk_ema is not None:
                brk_ema = args.brk_alpha * brk + (1.0 - args.brk_alpha) * brk_ema
                brk = clamp(brk_ema, 0.0, 1.0)

            steer = clamp(steer_raw, -1.0, 1.0)
            if abs(steer) < args.steer_deadband: steer = 0.0
            if steer_ema is not None:
                steer_ema = args.steer_alpha * steer + (1.0 - args.steer_alpha) * steer_ema
                steer = clamp(steer_ema, -1.0, 1.0)
            if args.steer_rate > 0.0 and dt and dt > 0.0:
                max_step = args.steer_rate * dt
                steer = clamp(steer, steer_prev - max_step, steer_prev + max_step)
            steer_prev = steer

            ego.apply_control(carla.VehicleControl(
                throttle=float(thr),
                brake=float(brk),
                steer=float(steer),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            ))

            now = time.time()
            if now - last_log >= max(0.1, args.log_every):
                v = speed_kmh(ego)
                print(f"[CTL] v={v:5.1f} km/h | thr={thr:4.2f} brk={brk:4.2f} steer={steer:+.3f}"
                      + (f" | mode={_mode}" if _mode is not None else ""))
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
