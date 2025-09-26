#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_lk_kuksa_added.py — LK + ACC (전부 Kuksa 경유), 카메라 입력은 Zenoh
- Env: CARLA 센서 → Kuksa(ACC 5종 센서), 카메라 → Zenoh
- Decision(본 파일): 
    * Zenoh에서 카메라 구독 → 차선검출 → LK 스티어 계산 후 Kuksa에 publish
    * Kuksa에서 ACC 센서 5종 읽기 → ACC 컨트롤 → Throttle/Brake/Mode를 Kuksa에 publish
- Control: Kuksa의 ACC/LK 명령 읽어 CARLA 적용 (별도 control.py)
"""

import os, time, json, argparse
import numpy as np
import cv2
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- LK, ACC 알고리즘 ---
from common.LK_algo import (
    detect_lanes_and_center, fuse_lanes_with_memory,
    lane_mid_x, lookahead_ratio, gains_for_speed
)
from common.acc_algo import ACCController, ACCInputs

# ================= Zenoh 세션 (전역) =================
cfg = zenoh.Config()
try:
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json("mode", '"client"')
    cfg.insert_json("connect/endpoints", '["tcp/127.0.0.1:7447"]')
z = zenoh.open(cfg)

# ================= KUKSA: LK 스티어 publish =================
_last_steer = 0.0  # 직전 값 재사용용
def publish_steer(kuksa: VSSClient, st):
    global _last_steer
    st = float(_last_steer if st is None else st)
    _last_steer = st
    # actuator지만, 컨트롤이 current를 읽는 경우도 있어서 current에 씀 (필요시 target으로 변경 가능)
    kuksa.set_target_values({"Vehicle.ADAS.LK.Steering": Datapoint(st)})

# ================= LK 유틸 =================
DEF_TOPIC_CAM = "carla/cam/front"

def parse_attachment(att: bytes) -> dict:
    try:
        return json.loads(att.decode("utf-8"))
    except Exception:
        return {}

def bgra_to_bgr(buf: memoryview, w: int, h: int) -> np.ndarray:
    return np.frombuffer(buf, np.uint8).reshape((h, w, 4))[:, :, :3].copy()

# ================= ACC: Kuksa 경로/유틸 =================
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

def _unwrap(x):
    """Datapoint/dict/list 어떤 형태든 value만 뽑기"""
    try:
        return getattr(x, "value")
    except Exception:
        pass
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    if isinstance(x, (list, tuple)) and x:
        return _unwrap(x[0])
    return x

def read_acc_sensors(kc: VSSClient):
    """Kuksa current에서 ACC 센서 5종 일괄 읽기(dict 반환)"""
    try:
        res = kc.get_current_values(ACC_SENS_PATHS)
        if not isinstance(res, dict):
            res = {p: kc.get_current_value(p) for p in ACC_SENS_PATHS}
    except AttributeError:
        res = {p: kc.get_current_value(p) for p in ACC_SENS_PATHS}
    return {k: _unwrap(v) for k, v in res.items()}

def write_acc_actuators(kc: VSSClient, thr: float, brk: float, mode: str):
    """ACC 액추에이터 쓰기: target 우선, 실패 시 current 폴백"""
    thr = float(max(0.0, min(1.0, thr)))
    brk = float(max(0.0, min(1.0, brk)))

    kc.set_target_values({
        ACC_THR_PATH:  Datapoint(thr),
        ACC_BRK_PATH:  Datapoint(brk),
        ACC_MODE_PATH: Datapoint(mode),
    })

# ================= ACC 호환 헬퍼(시그니처/필드명 차이 흡수) =================
from types import SimpleNamespace

def _acc_build_inputs(v_set_mps, **kwargs):
    """
    ACCInputs 생성 시 v_set 인자 지원/미지원 모두 커버:
    - 우선 v_set 키워드로 생성 시도
    - 안되면 ACCInputs(**kwargs) 후 적절한 속성명(v_set, v_set_mps, target_speed, ...)을 찾아 주입
    - 그것도 안되면 SimpleNamespace로 대체
    """
    try:
        return ACCInputs(**kwargs, v_set=v_set_mps)
    except TypeError:
        try:
            obj = ACCInputs(**kwargs)
            for name in ("v_set", "v_set_mps", "target_speed", "target_speed_mps", "setpoint"):
                if hasattr(obj, name):
                    setattr(obj, name, v_set_mps)
                    break
            return obj
        except Exception:
            return SimpleNamespace(**kwargs, v_set=v_set_mps)
        
import inspect
from types import SimpleNamespace

def _acc_step(ctrl, inputs, dt, v_set_mps):
    """
    ACCController.step 시그니처 차이를 흡수:
    1) 파라미터 이름을 기준으로 kwargs 매핑 시도
    2) 실패 시 안전한 순열들을 차례대로 시도
    3) 전부 실패하면 안전한 기본값 반환
    """
    # 1) 이름 기반 매핑 시도
    try:
        sig = inspect.signature(ctrl.step)
        params = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        # self 제거
        names = [p.name for p in params if p.name != "self"]
        if names:
            kw = {}
            for name in names:
                n = name.lower()
                if n in ("inputs", "inp", "acc_in", "meas", "measurement", "state"):
                    kw[name] = inputs
                elif n in ("dt", "delta_t", "delta", "delta_time", "step", "ts", "timestep"):
                    kw[name] = dt
                elif n in ("v_set", "vset", "target_speed", "target_speed_mps",
                           "setpoint", "v_ref", "v_target", "v_des", "v_des_mps"):
                    kw[name] = v_set_mps
                # 혹시 모를 이름들에 대한 관용 매핑
                elif n in ("time_step", "time_dt"):
                    kw[name] = dt
                elif n in ("set_speed", "cruise", "cruise_speed"):
                    kw[name] = v_set_mps
            # 이름이 전혀 안 맞으면 kwargs 호출을 생략
            if kw:
                return ctrl.step(**kw)
    except Exception:
        pass

    # 2) 위치 인자 순열 시도 (inputs / dt / v_set_mps 순서가 뒤섞였을 가능성)
    tries = (
        lambda: ctrl.step(inputs, dt, v_set_mps),
        lambda: ctrl.step(inputs, v_set_mps, dt),
        lambda: ctrl.step(v_set_mps, dt, inputs),
        lambda: ctrl.step(dt, inputs, v_set_mps),
        lambda: ctrl.step(v_set_mps, inputs, dt),
        lambda: ctrl.step(dt, v_set_mps, inputs),
        # 2-arg 패턴
        lambda: ctrl.step(inputs, dt),
        lambda: ctrl.step(inputs, v_set_mps),
        lambda: ctrl.step(v_set_mps, inputs),
        lambda: ctrl.step(dt, inputs),
        lambda: ctrl.step(dt, v_set_mps),
        lambda: ctrl.step(v_set_mps, dt),
        # 1-arg 패턴 (컨트롤러가 내부 상태/세트포인트 들고 있을 때)
        lambda: ctrl.step(inputs),
        lambda: ctrl.step(dt),
        lambda: ctrl.step(v_set_mps),
        # kwargs 혼합 패턴
        lambda: ctrl.step(inputs=inputs, dt=dt),
        lambda: ctrl.step(inputs=inputs, v_set=v_set_mps),
        lambda: ctrl.step(dt=dt, v_set=v_set_mps),
        lambda: ctrl.step(inputs=inputs, dt=dt, v_set=v_set_mps),
    )
    last_err = None
    for f in tries:
        try:
            return f()
        except Exception as e:
            last_err = e
            continue

    # 3) 전부 실패 → 경고 후 안전값
    if last_err:
        print("[ACC] WARN:", repr(last_err))
    return SimpleNamespace(throttle=0.0, brake=0.0, mode="ERR_SIG")

def _acc_unpack(out):
    """반환(객체/dict)의 필드명 차이 흡수"""
    if isinstance(out, dict):
        thr = float(out.get("throttle", out.get("acc_throttle", 0.0)))
        brk = float(out.get("brake",    out.get("acc_brake",    0.0)))
        mode = str(out.get("mode", out.get("state", "UNKNOWN")))
        return thr, brk, mode
    thr = float(getattr(out, "throttle", getattr(out, "acc_throttle", 0.0)))
    brk = float(getattr(out, "brake",    getattr(out, "acc_brake",    0.0)))
    mode = str(getattr(out, "mode",      getattr(out, "state", "UNKNOWN")))
    return thr, brk, mode

# ================= 메인 =================
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
    ap.add_argument("--roi_json", type=str, default="")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--canny_low", type=int, default=60)
    ap.add_argument("--canny_high", type=int, default=180)
    ap.add_argument("--hough_thresh", type=int, default=40)
    ap.add_argument("--hough_min_len", type=int, default=20)
    ap.add_argument("--hough_max_gap", type=int, default=60)
    ap.add_argument("--lane_mem_ttl", type=int, default=90)
    ap.add_argument("--anchor_speed_kmh", type=float, default=50.0)
    # ACC
    ap.add_argument("--acc_target_kmh", type=float, default=30.0, help="ACC 크루즈 목표 속도")
    ap.add_argument("--acc_print", type=int, default=1)
    # Viz/Log
    ap.add_argument("--display", type=int, default=1)
    ap.add_argument("--print", type=int, default=1)
    args = ap.parse_args()
    dt = 1.0 / max(1, args.fps)

    # Kuksa 연결
    kuksa = VSSClient(args.kuksa_host, args.kuksa_port)
    kuksa.connect()
    print(f"[decision_LK+ACC] Kuksa connected @ {args.kuksa_host}:{args.kuksa_port}")

    # Zenoh 카메라 구독
    latest = {"bgr": None, "meta": {}, "ts": 0.0}
    no_frame_tick = 0

    def _on_cam(sample):
        try:
            # payload
            raw = None
            if hasattr(sample, "payload"):
                raw = bytes(sample.payload)
            elif hasattr(sample, "value") and hasattr(sample.value, "payload"):
                raw = bytes(sample.value.payload)
            if raw is None:
                return
            # meta(attachment)
            meta = None
            att_bytes = getattr(sample, "attachment", None)
            if att_bytes:
                try:
                    meta = json.loads(att_bytes.decode("utf-8"))
                except Exception:
                    meta = None
            # w/h
            if meta and "w" in meta and "h" in meta:
                w = int(meta["w"]); h = int(meta["h"])
            else:
                w = int(args.width); h = int(args.height)
            # BGRA → BGR
            if len(raw) != w * h * 4:
                if args.print:
                    print(f"[WARN] bad payload size: got={len(raw)} expected={w*h*4} (w={w},h={h})")
                return
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

    # ACC 컨트롤러 & setpoint
    acc_ctrl = ACCController()
    v_set = float(args.acc_target_kmh) / 3.6  # m/s

    print("[INFO] decision_LK+ACC running. Press q to quit.")
    try:
        next_t = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            # 프레임 대기
            if latest["bgr"] is None:
                no_frame_tick += 1
                if args.print and (no_frame_tick % max(1, args.fps) == 0):
                    print("[WAIT] no camera frame yet... "
                          f"topic='{args.topic_cam}', endpoint='{args.zenoh_endpoint}'")
                if args.display:
                    cv2.waitKey(1)
                continue

            no_frame_tick = 0
            bgr = latest["bgr"]
            h, w = bgr.shape[:2]
            frame_id = int(latest["meta"].get("frame", 0))

            # ===== LK: 차선검출 & 조향 =====
            lanes = detect_lanes_and_center(
                bgr,
                roi_vertices=None,
                canny_low=args.canny_low, canny_high=args.canny_high,
                hough_thresh=args.hough_thresh,
                hough_min_len=args.hough_min_len,
                hough_max_gap=args.hough_max_gap,
            )
            used_left, used_right, center_line, _center_x = fuse_lanes_with_memory(
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
                v_kmh = 0.0
                kp, st_clip = gains_for_speed(v_kmh / 3.6)
                st = float(max(-st_clip, min(st_clip, kp * offset_px)))
            else:
                st = 0.0

            publish_steer(kuksa, st)   # LK 스티어 Kuksa publish

            # ===== ACC: Kuksa 센서 읽기 → 컨트롤 → 액추에이터 쓰기 =====
            acc_thr, acc_brk, acc_mode = 0.0, 0.0, "IDLE"
            try:
                sens = read_acc_sensors(kuksa)
                d   = sens.get("Vehicle.ADAS.ACC.Distance", None)
                rv  = sens.get("Vehicle.ADAS.ACC.RelSpeed", None)
                ttc = sens.get("Vehicle.ADAS.ACC.TTC", None)
                ht  = bool(sens.get("Vehicle.ADAS.ACC.HasTarget", False))
                vL  = sens.get("Vehicle.ADAS.ACC.LeadSpeedEst", None)

                acc_in = _acc_build_inputs(
                    v_set_mps=v_set,
                    distance=d, rel_speed=rv, ttc=ttc,
                    has_target=ht, lead_speed_est=vL
                )
                acc_out = _acc_step(acc_ctrl, acc_in, dt, v_set)
                acc_thr, acc_brk, acc_mode = _acc_unpack(acc_out)
                write_acc_actuators(kuksa, acc_thr, acc_brk, acc_mode)

                if args.acc_print:
                    print(f"[ACC] d={d} rv={rv} ttc={ttc} ht={ht} v_set={v_set*3.6:.0f} "
                          f"-> thr={acc_thr:.2f} brk={acc_brk:.2f} mode={acc_mode}")
            except Exception as e:
                if args.acc_print:
                    print("[ACC] WARN:", repr(e))

            # ===== 로그 & 시각화 =====
            if args.print:
                print(f"v={0.0:5.1f} km/h  thr={acc_thr:.2f} brk={acc_brk:.2f} str={st:+.2f}")

            if args.display:
                vis = bgr.copy()
                rvx = lanes.get("roi_vertices")
                if rvx is not None:
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, rvx, (0, 0, 255))
                    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0.0)
                for x1, y1, x2, y2 in lanes.get("line_segs", []):
                    cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)
                if used_left:
                    cv2.line(vis, (used_left[0], used_left[1]),
                             (used_left[2], used_left[3]), (0, 0, 255), 5, cv2.LINE_AA)
                if used_right:
                    cv2.line(vis, (used_right[0], used_right[1]),
                             (used_right[2], used_right[3]), (0, 0, 255), 5, cv2.LINE_AA)
                cv2.line(vis, (x_cam_mid, 0), (x_cam_mid, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
                if x_lane_mid is not None:
                    cv2.line(vis, (x_lane_mid, 0), (x_lane_mid, h - 1), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(vis, (0, y_anchor), (w - 1, y_anchor), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, f"thr={acc_thr:.2f} brk={acc_brk:.2f} str={st:+.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("LK_decision", vis)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        try: kuksa.close()
        except Exception:
            try: kuksa.disconnect()
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
