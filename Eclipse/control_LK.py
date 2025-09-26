#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import argparse
import traceback
import carla
import numpy as np
from typing import Any, Optional
from queue import Queue

from kuksa_client.grpc import VSSClient, Datapoint

# --- LK, ACC 알고리즘 호출 ---
# =============================================================
from common.LK_algo import (speed_of)
# =============================================================

# --- 유틸 ---
# =============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def to_float_safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def find_actor_by_role(world: carla.World, role_name: str) -> Optional[carla.Actor]:
    for actor in world.get_actors().filter("*vehicle*"):
        if actor.attributes.get("role_name") == role_name:
            return actor
    return None
# =============================================================

# ---------- Main ----------
def main():
    # --- Argparse ---
    ap = argparse.ArgumentParser("control_LK (sync + sleep pacing)")
    # CARLA / TM
    ap.add_argument("--carla_host", default="127.0.0.1")
    ap.add_argument("--carla_port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)

    # 로그
    ap.add_argument("--log_every", type=float, default=0.5)

    # 스로틀(P) + EMA
    ap.add_argument("--target_speed_kmh", type=float, default=50.0)
    ap.add_argument("--kp_throttle", type=float, default=0.05)
    ap.add_argument("--ema_alpha_speed", type=float, default=0.2)
    ap.add_argument("--ema_alpha_thr", type=float, default=0.25)

    # 조향
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)
    ap.add_argument("--steer_key", default="Vehicle.ADAS.LK.Steering")
    ap.add_argument("--steer_alpha", type=float, default=0.4, help="0이면 필터 미사용")
    ap.add_argument("--deadband", type=float, default=0.03,
                help="조향 deadband (normed steer 절댓값이 이 값보다 작으면 0으로 처리)")
    # ---- 추가 ----
    ap.add_argument("--ki_steer", type=float, default=0.02, help="조향 적분 게인(작게!)")
    ap.add_argument("--i_clip", type=float, default=0.20, help="조향 적분 한계(anti-windup)")
    ap.add_argument("--i_decay_tau", type=float, default=2.0, help="중심 근처에서 적분 감쇠 시간상수[s]")

    # 페이싱(수면) 옵션
    ap.add_argument("--extra_sleep_ratio", type=float, default=0.12,
                    help="0~0.3 권장: dt에서 처리시간을 뺀 뒤 추가로 ratio*dt만큼 더 잠(시각적 안정성↑)")

    args = ap.parse_args()

    # --- CARLA 연결 / 액터 ---
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world: carla.World = client.get_world()
    settings = world.get_settings()
    print(f"[WORLD] connected. sync={settings.synchronous_mode}, fixed_dt={settings.fixed_delta_seconds}")

    if not settings.synchronous_mode:
        print("[WARN] World is NOT in synchronous_mode. Env에서 sync 적용이 필요합니다.")
    if not isinstance(settings.fixed_delta_seconds, (float, int)) or settings.fixed_delta_seconds is None:
        print("[WARN] fixed_delta_seconds가 설정되지 않았습니다(Env에서 설정 권장).")

    ego = find_actor_by_role(world, "ego")
    lead = find_actor_by_role(world, "lead")
    if not ego:
        raise RuntimeError("EGO (role_name='ego') not found")
    if not lead:
        raise RuntimeError("LEAD (role_name='lead') not found")
    print(f"[INFO] ego id={ego.id}, lead id={lead.id}")

    # --- 역할 설정: ego 수동, lead TM ON ---
    ego.set_autopilot(False)
    tm = client.get_trafficmanager(args.tm_port)
    lead.set_autopilot(True, args.tm_port)
    # 예: 제한속도보다 60% 느리게(= 제한속도의 40%로 주행)
    tm.vehicle_percentage_speed_difference(lead, 60.0)

    try:
        tm.set_synchronous_mode(True)  # 월드가 sync일 때 의미 있음
    except Exception:
        pass
    try:
        tm.auto_lane_change(lead, False)
    except Exception:
        pass
    print(f"[INFO] Lead autopilot ON via TM:{args.tm_port}")

    # --- KUKSA 연결 ---
    kuksa = VSSClient(args.kuksa_host, args.kuksa_port)
    kuksa.connect()
    print(f"[KUKSA] connected @ {args.kuksa_host}:{args.kuksa_port}")

    # --- Env tick 동기 큐 ---
    tick_q: Queue = Queue(maxsize=2)
    world.on_tick(tick_q.put)

    # --- 상태 변수 ---
    ema_v = None
    ema_thr = 0.0
    steer_ema = None if args.steer_alpha <= 0.0 else 0.0
    last_log = time.time()
    i_state = 0.0  # 적분 상태(초기 0)

    print(f"[RUN] sync+sleep | target_v≈{args.target_speed_kmh:.1f} km/h | steer key='{args.steer_key}'")

    try:
        while True:
            # 1) 틱 동기 대기 (Env가 world.tick() 호출해야 여기 들어옴)
            snap: carla.Timestamp = tick_q.get()
            dt = snap.delta_seconds if (snap and snap.delta_seconds) else (settings.fixed_delta_seconds or 0.05)
            t0 = time.time()

            # 2) 속도 읽기 + EMA
            v_now = speed_of(ego)

            # 3) 스로틀 P + EMA
            throttle = 0.5

            # 4) 조향: dict → Datapoint.value 가정
            resp = kuksa.get_current_values([args.steer_key])
            steer_raw = float(resp[args.steer_key].value)
            # steer_raw = clamp(steer_raw, -1.0, 1.0)
            if abs(steer_raw) < args.deadband:
                steer_raw = 0.0

            # 옵션 EMA(프레임 기준)
            if steer_ema is not None:
                steer_ema = args.steer_alpha * steer_raw + (1.0 - args.steer_alpha) * steer_ema
                steer_out = clamp(steer_ema, -1.0, 1.0)
            else:
                steer_out = steer_raw

            # --- 추가 ---
            decay = 0.0
            if args.i_decay_tau > 1e-6:
                # 작은 명령에서는 감쇠 강하게, 큰 명령에서는 거의 감쇠 안 함
                # (|steer_out|이 0이면 e^-0=1 → 감쇠 빠름, 0.3이면 e^-(0.3/τ))
                decay = math.exp(-abs(steer_out) / max(args.i_decay_tau, 1e-6))

            # 적분 업데이트: 중심 근처일수록 i_state를 자연감쇠, 그 외엔 누적
            i_state = i_state * decay + args.ki_steer * steer_raw * dt

            # anti-windup
            i_state = clamp(i_state, -args.i_clip, args.i_clip)

            # 최종 명령: P(=steer_out) + I
            steer_cmd = clamp(steer_out + i_state, -1.0, 1.0)

            # 5) 적용 (한 틱당 정확히 1회)
            ego.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer_out),
                brake=0.0
            ))

            # 6) 로그(벽시계 간격)
            now = time.time()
            if (now - last_log) >= args.log_every:
                print(f"[CTL] dt={dt:.3f}s | v={v_now:5.1f} km/h | thr={throttle:4.2f} | steer={steer_out:+.3f}")
                last_log = now

            # 7) 수면(sleep) 기반 페이싱: 틱 처리시간을 고려해 잔여 시간 + 여유 sleep
            proc_dt = time.time() - t0
            target_dt = float(dt if dt and dt > 0 else (settings.fixed_delta_seconds or 0.05))
            extra = args.extra_sleep_ratio * target_dt
            to_sleep = (target_dt - proc_dt) + extra
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    except Exception as e:
        print("[ERR]", e)
        traceback.print_exc()
    finally:
        try:
            ego.set_autopilot(False)
        except Exception:
            pass
        try:
            kuksa.close()
        except Exception:
            pass
        print("[CLEAN] control_LK exit")


if __name__ == "__main__":
    main()