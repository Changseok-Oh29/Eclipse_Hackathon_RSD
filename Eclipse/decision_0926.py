#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
decision_lk_acc_kuksa.py — LK + ACC (Kuksa/Zenoh)
- 카메라(zenoh)로 차선 검출 → LK 스티어 계산 후 Kuksa에 publish
- Kuksa의 ACC 센서(거리/상대속도/ttc/타깃유무/리드추정속도)를 읽어 ACC 출력(thr/brk/mode) 계산 후 Kuksa에 publish
- publish는 dual-write(target + current)로 작성해 어떤 control도 값을 읽도록 보장
- CARLA RADAR 부호 보정(+는 멀어짐) → rv_sign=-1.0 권장
- 접근일 때만 TTC 유한값, 아니면 ∞로 정리. 유효 타깃(eff_ht) 재판정.
- eff_ht=False(실질 타깃 없음)일 때는 cruise_thr_floor로 킥 없이도 출발
"""

import os, time, json, argparse, inspect
import numpy as np
import cv2
import zenoh
from types import SimpleNamespace
from kuksa_client.grpc import VSSClient, Datapoint
import common.acc_algo as ACCP  # <- 모듈 자체도 import


os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- LK, ACC 알고리즘 ---
from common.LK_algo import (
    detect_lanes_and_center, fuse_lanes_with_memory,
    lane_mid_x, lookahead_ratio, gains_for_speed
)
from common.acc_algo import ACCController, ACCInputs

DEF_TOPIC_CAM = "carla/cam/front"
ACC_SENS_PATHS = [
    "Vehicle.ADAS.ACC.Distance",
    "Vehicle.ADAS.ACC.RelSpeed",
    "Vehicle.ADAS.ACC.TTC",
    "Vehicle.ADAS.ACC.HasTarget",
    "Vehicle.ADAS.ACC.LeadSpeedEst",
]
ACC_THR_PATH  = "Vehicle.ADAS.ACC.Ctrl.Throttle"
ACC_BRK_PATH  = "Vehicle.ADAS.ACC.Ctrl.Brake"
ACC_MODE_PATH = "Vehicle.ADAS.ACC.Ctrl.Mode"
LK_STEER_PATH = "Vehicle.ADAS.LK.Steering"

# ---------- Kuksa helpers ----------
def _unwrap(x):
    try: return getattr(x, "value")
    except Exception: pass
    if isinstance(x, dict) and "value" in x: return x["value"]
    if isinstance(x, (list, tuple)) and x:  return _unwrap(x[0])
    return x

def _as_bool(x):
    if isinstance(x, bool): return x
    try:
        if isinstance(x, (int,float)): return x != 0
        if isinstance(x, str): return x.strip().lower() in ("1","true","t","yes","y","on")
    except: pass
    return False

def read_acc_sensors(kc: VSSClient):
    try:
        res = kc.get_current_values(ACC_SENS_PATHS)
        if not isinstance(res, dict):
            res = {p: kc.get_current_value(p) for p in ACC_SENS_PATHS}
    except AttributeError:
        res = {p: kc.get_current_value(p) for p in ACC_SENS_PATHS}
    return {k: _unwrap(v) for k, v in res.items()}

# dual-write (target + current)
def publish_steer_dual(kuksa: VSSClient, st: float):
    st = float(st)
    try: kuksa.set_target_values({LK_STEER_PATH: Datapoint(st)})
    except Exception: pass
    try: kuksa.set_current_values({LK_STEER_PATH: Datapoint(st)})
    except Exception: pass

def write_acc_dual(kc: VSSClient, thr: float, brk: float, mode: str):
    thr = float(max(0.0, min(1.0, thr)))
    brk = float(max(0.0, min(1.0, brk)))
    try:
        kc.set_target_values({
            ACC_THR_PATH:  Datapoint(thr),
            ACC_BRK_PATH:  Datapoint(brk),
            ACC_MODE_PATH: Datapoint(mode),
        })
    except Exception: pass
    try:
        kc.set_current_values({
            ACC_THR_PATH:  Datapoint(thr),
            ACC_BRK_PATH:  Datapoint(brk),
            ACC_MODE_PATH: Datapoint(mode),
        })
    except Exception: pass

# ---------- ACC signature adapters ----------
def _acc_build_inputs(v_set_mps, **kwargs):
    try:
        return ACCInputs(**kwargs, v_set=v_set_mps)  # 일부 구현체 호환
    except TypeError:
        try:
            obj = ACCInputs(**kwargs)
            for name in ("v_set","v_set_mps","target_speed","target_speed_mps","setpoint"):
                if hasattr(obj, name):
                    setattr(obj, name, v_set_mps)
                    break
            return obj
        except Exception:
            return SimpleNamespace(**kwargs, v_set=v_set_mps)

def _acc_step(ctrl, acc_in, dt, v_set_mps):
    """
    다양한 ACCController.step 시그니처 지원:
    - (sim_time, ego_speed_mps, acc_inputs)  ← 사용자가 준 acc_algo.py
    - (inputs, dt, v_set) 등 기타 변종
    ego_speed_mps는 decision에서 모름 → 0.0으로 보냄(크루즈 가속 유도)
    """
    try:
        sig = inspect.signature(ctrl.step)
        names = [p.name for p in sig.parameters.values() if p.name != "self"]
        if names == ["sim_time", "ego_speed_mps", "acc"]:
            return ctrl.step(time.perf_counter(), 0.0, acc_in)  # ego v unknown → 0.0
        # 이름 기반 추정
        kw = {}
        for name in names:
            n = name.lower()
            if n in ("inputs","inp","acc","acc_in","meas","measurement","state"):
                kw[name] = acc_in
            elif n in ("dt","delta_t","delta","delta_time","step","ts","timestep","time_step","time_dt"):
                kw[name] = dt
            elif n in ("v_set","vset","target_speed","target_speed_mps","setpoint",
                       "v_ref","v_target","v_des","v_des_mps","set_speed","cruise","cruise_speed"):
                kw[name] = v_set_mps
            elif n in ("sim_time","time_s","t_now"): kw[name] = time.perf_counter()
            elif n in ("ego_speed","ego_speed_mps","v_ego"): kw[name] = 0.0
        if kw:
            return ctrl.step(**kw)
    except Exception:
        pass

    # fallbacks
    tries = (
        lambda: ctrl.step(time.perf_counter(), 0.0, acc_in),
        lambda: ctrl.step(acc_in, dt, v_set_mps),
        lambda: ctrl.step(acc_in, v_set_mps, dt),
        lambda: ctrl.step(v_set_mps, dt, acc_in),
        lambda: ctrl.step(dt, acc_in, v_set_mps),
        lambda: ctrl.step(dt, v_set_mps, acc_in),
        lambda: ctrl.step(acc_in),
    )
    for f in tries:
        try: return f()
        except Exception: pass
    return SimpleNamespace(throttle=0.0, brake=0.0, mode="ERR_SIG")

def _acc_unpack(out):
    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            thr = float(out[0])
            brk = float(out[1])
            mode = str(out[2])
            return thr, brk, mode
        # 안전 폴백
        return 0.0, 0.0, "UNKNOWN"
    if isinstance(out, dict):
        thr = float(out.get("throttle", out.get("acc_throttle", 0.0)))
        brk = float(out.get("brake",    out.get("acc_brake",    0.0)))
        mode = str(out.get("mode", out.get("state", "UNKNOWN")))
        return thr, brk, mode
    thr = float(getattr(out, "throttle", getattr(out, "acc_throttle", 0.0)))
    brk = float(getattr(out, "brake",    getattr(out, "acc_brake",    0.0)))
    mode = str(getattr(out, "mode",      getattr(out, "state", "UNKNOWN")))
    return thr, brk, mode

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # Zenoh
    ap.add_argument("--zenoh_endpoint", default="tcp/127.0.0.1:7447")
    ap.add_argument("--topic_cam", default=DEF_TOPIC_CAM)
    ap.add_argument("--fps", type=int, default=20)
    # Kuksa
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)
    # LK
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--canny_low", type=int, default=60)
    ap.add_argument("--canny_high", type=int, default=180)
    ap.add_argument("--hough_thresh", type=int, default=40)
    ap.add_argument("--hough_min_len", type=int, default=20)
    ap.add_argument("--hough_max_gap", type=int, default=60)
    ap.add_argument("--lane_mem_ttl", type=int, default=90)
    ap.add_argument("--anchor_speed_kmh", type=float, default=50.0)
    # ACC 출발/게이팅
    ap.add_argument("--acc_target_kmh", type=float, default=70.0)
    ap.add_argument("--rv_sign", type=float, default=-1.0, help="상대속도 부호 보정(CARLA RADAR: +는 멀어짐) → -1")
    ap.add_argument("--launch_ttc_s", type=float, default=8.0, help="TTC가 이 값보다 크면 타깃 무시(출발 허용)")
    ap.add_argument("--launch_gap_m", type=float, default=10.0, help="거리 d가 이 값보다 크면 타깃 무시")
    ap.add_argument("--cruise_thr_floor", type=float, default=1, help="타깃 무시 상태 최소 추진력")

    ap.add_argument("--rv_ema", type=float, default=0.40)          # 상대속도 저역통과
    ap.add_argument("--ttc_closing_floor", type=float, default=0.4)  # TTC 분모 최소 m/s
    ap.add_argument("--follow_gate_gap_m", type=float, default=80.0)  # 이 거리 이내면 FOLLOW 후보
    ap.add_argument("--follow_gate_ttc_s", type=float, default=25.0)  # 이 TTC 이하면 FOLLOW 후보
    ap.add_argument("--acc_follow_enter_ttc", type=float, default=14.0)  # ACC 내부 FOLLOW 진입
    ap.add_argument("--acc_follow_exit_ttc",  type=float, default=16.0)  # ACC 내부 FOLLOW 이탈

    ap.add_argument("--acc_print", type=int, default=1)
    # Viz/Log
    ap.add_argument("--display", type=int, default=1)
    ap.add_argument("--print", type=int, default=1)
    args = ap.parse_args()
    dt = 1.0 / max(1, args.fps)

    # Kuksa
    kuksa = VSSClient(args.kuksa_host, args.kuksa_port)
    kuksa.connect()
    print(f"[decision_LK+ACC] Kuksa connected @ {args.kuksa_host}:{args.kuksa_port}")

    # Zenoh
    zcfg = zenoh.Config()
    try:
        zcfg.insert_json5("mode", '"client"')
        zcfg.insert_json5("connect/endpoints", f'["{args.zenoh_endpoint}"]')
    except AttributeError:
        zcfg.insert_json("mode", '"client"')
        zcfg.insert_json("connect/endpoints", f'["{args.zenoh_endpoint}"]')
    z = zenoh.open(zcfg)

    latest = {"bgr": None, "meta": {}, "ts": 0.0}
    def _on_cam(sample):
        try:
            raw = None
            if hasattr(sample, "payload"): raw = bytes(sample.payload)
            elif hasattr(sample, "value") and hasattr(sample.value, "payload"):
                raw = bytes(sample.value.payload)
            if raw is None: return
            att = getattr(sample, "attachment", None)
            meta = None
            if att:
                try: meta = json.loads(att.decode("utf-8"))
                except Exception: meta = None
            w = int((meta or {}).get("w", args.width))
            h = int((meta or {}).get("h", args.height))
            if len(raw) != w*h*4: return
            arr = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            latest["bgr"] = arr[:, :, :3].copy()
            latest["meta"] = meta or {"w": w, "h": h, "frame": -1}
            latest["ts"] = time.time()
        except Exception as e:
            print("[ERR] cam callback:", repr(e))
    sub_cam = z.declare_subscriber(args.topic_cam, _on_cam)

    if args.display:
        cv2.namedWindow("LK_decision", cv2.WINDOW_NORMAL)

    lane_mem = {"left": None, "right": None, "t_left": None, "t_right": None, "center_x": None}
    acc_ctrl = ACCController()
    try:
        ACCP.FOLLOW_ENTER_TTC = float(args.acc_follow_enter_ttc)
        ACCP.FOLLOW_EXIT_TTC  = float(args.acc_follow_exit_ttc)
    except Exception:
        pass
    v_set = float(args.acc_target_kmh) / 3.6

    print("[INFO] decision_LK+ACC running. Press q to quit.")
    try:
        next_t = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now < next_t: time.sleep(next_t - now)
            next_t += dt

            if latest["bgr"] is None:
                if args.display: cv2.waitKey(1)
                continue

            # ================= LK =================
            bgr = latest["bgr"]
            h, w = bgr.shape[:2]
            frame_id = int(latest["meta"].get("frame", 0))
            lanes = detect_lanes_and_center(
                bgr, roi_vertices=None,
                canny_low=args.canny_low, canny_high=args.canny_high,
                hough_thresh=args.hough_thresh,
                hough_min_len=args.hough_min_len,
                hough_max_gap=args.hough_max_gap,
            )
            used_left, used_right, _, _ = fuse_lanes_with_memory(
                lanes, frame_id, lane_mem, ttl_frames=args.lane_mem_ttl
            )
            y_anchor = int(lookahead_ratio(args.anchor_speed_kmh) * h)
            x_cam_mid = w // 2
            x_lane_mid = lane_mem.get("center_x")
            if used_left and used_right:
                x_lane_mid, _, _ = lane_mid_x(used_left, used_right, y_anchor)
                lane_mem["center_x"] = x_lane_mid

            st = 0.0
            if x_lane_mid is not None:
                offset_px = (x_lane_mid - x_cam_mid)
                kp, st_clip = gains_for_speed(0.0)
                st = float(max(-st_clip, min(st_clip, kp * offset_px)))
            publish_steer_dual(kuksa, st)

            # ================= ACC =================
            sens = read_acc_sensors(kuksa)
            d_raw  = sens.get("Vehicle.ADAS.ACC.Distance", None)
            rv_raw = sens.get("Vehicle.ADAS.ACC.RelSpeed", 0.0)   # CARLA: + 멀어짐
            ttc_raw= sens.get("Vehicle.ADAS.ACC.TTC", None)
            ht_raw = _as_bool(sens.get("Vehicle.ADAS.ACC.HasTarget", 0))
            vL     = sens.get("Vehicle.ADAS.ACC.LeadSpeedEst", None)

            # 부호 보정(+면 '접근')
            # --- 상대속도(+면 '접근'이 되도록 부호 보정) ---
            rv_app = float(args.rv_sign) * float(rv_raw or 0.0)

            # (선택) 간단 EMA로 rv 안정화
            if not hasattr(main, "_rv_f"):
                main._rv_f = rv_app
            rv_alpha = 0.45  # 0.35~0.55 권장
            main._rv_f = rv_alpha * rv_app + (1.0 - rv_alpha) * main._rv_f

            # --- TTC: 접근이 아주 작아도 ∞만 나오지 않게 분모에 floor 적용 ---
            closing = max(0.0, main._rv_f)     # 접근 성분만
            d_val = float(d_raw) if d_raw is not None else float("inf")
            closing_floor = 0.30               # m/s (0.25~0.5 권장)
            denom = max(closing, closing_floor)
            ttc_sane = (d_val / denom) if np.isfinite(d_val) else float("inf")

            # --- 유효 타깃 판정: 거리/ttc 둘 중 하나라도 빠르면 FOLLOW 후보 ---
            DIST_STRICT = 35.0    # 이내면 HasTarget 불안정해도 무조건 타깃 인정
            FOLLOW_GAP_GATE = 80.0
            FOLLOW_TTC_GATE = 30.0

            eff_ht = (
                (np.isfinite(d_val) and d_val < DIST_STRICT) or
                (bool(ht_raw) and np.isfinite(d_val) and (
                    d_val < FOLLOW_GAP_GATE or (np.isfinite(ttc_sane) and ttc_sane < FOLLOW_TTC_GATE)
                ))
            )


            acc_in = _acc_build_inputs(
                v_set_mps=v_set,
                distance=(d_val if eff_ht else None),
                rel_speed=rv_app,                       # +면 접근
                ttc=(ttc_sane if eff_ht else float("inf")),
                has_target=eff_ht,
                lead_speed_est=(vL if eff_ht else None),
            )
            acc_out = _acc_step(acc_ctrl, acc_in, dt, v_set)
            acc_thr, acc_brk, acc_mode = _acc_unpack(acc_out)

            # eff_ht=False → 킥 없이도 천천히 출발
            if not eff_ht:
                if acc_thr < args.cruise_thr_floor:
                    acc_thr = args.cruise_thr_floor
                if acc_mode in ("UNKNOWN", "UNK", "IDLE", "ERR_SIG"):
                    acc_mode = "CRUISE"

            write_acc_dual(kuksa, acc_thr, acc_brk, acc_mode)

            if args.acc_print:
                ttc_print = f"{ttc_sane:.1f}" if np.isfinite(ttc_sane) else "inf"
                print(f"[ACC] d={None if d_raw is None else f'{d_val:.2f}'} "
                      f"rv_app={rv_app:+.3f} ttc={ttc_print} ht={eff_ht} "
                      f"v_set={v_set*3.6:.0f} -> thr={acc_thr:.2f} brk={acc_brk:.2f} mode={acc_mode}")

            # ================= Viz =================
            if args.display:
                vis = bgr.copy()
                for x1, y1, x2, y2 in lanes.get("line_segs", []):
                    cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)
                if used_left:
                    cv2.line(vis, (used_left[0], used_left[1]), (used_left[2], used_left[3]), (0, 0, 255), 5, cv2.LINE_AA)
                if used_right:
                    cv2.line(vis, (used_right[0], used_right[1]), (used_right[2], used_right[3]), (0, 0, 255), 5, cv2.LINE_AA)
                cv2.line(vis, (x_cam_mid, 0), (x_cam_mid, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
                if lane_mem.get("center_x") is not None:
                    cv2.line(vis, (lane_mem["center_x"], 0), (lane_mem["center_x"], h - 1), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(vis, (0, int(lookahead_ratio(args.anchor_speed_kmh) * h)), (w - 1, int(lookahead_ratio(args.anchor_speed_kmh) * h)), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, f"thr={acc_thr:.2f} brk={acc_brk:.2f} str={st:+.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("LK_decision", vis)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        try: kuksa.close()
        except Exception: pass
        try: sub_cam.undeclare()
        except Exception as e: print("[WARN] sub_cam.undeclare:", e)
        try: z.close()
        except Exception as e: print("[WARN] zenoh.close:", e)
        if args.display:
            try: cv2.destroyAllWindows()
            except Exception as e: print("[WARN] destroyAllWindows:", e)
        print("[INFO] decision.py stopped.")

if __name__ == "__main__":
    main()
