import os, time, json, argparse
import numpy as np
import cv2
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- LK, ACC 알고리즘 호출 ---
# =============================================================
from common.LK_algo import (
    detect_lanes_and_center, fuse_lanes_with_memory,
    lane_mid_x, lookahead_ratio, gains_for_speed
)
from common.acc_algo import ACCController, ACCInputs
# =============================================================


# --- Zenoh 환경 구축 ---
# =============================================================
cfg = zenoh.Config()
try:
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", f'["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json("mode", '"client"')
    cfg.insert_json("connect/endpoints", f'["tcp/127.0.0.1:7447"]')

z = zenoh.open(cfg)
# ==============================================================


# --- KUKSA 관련 wrapper ---
"""
decision이 매 프레임 호출.
st 계산하여 kuksa로 publish
"""
# =================================================================
_last_steer = 0.0  # 직전 값 재사용용

def publish_steer(kuksa: VSSClient, st):
    global _last_steer
    if st is None:
        st = _last_steer
    else:
        _last_steer = float(st)

    updates = {
        "Vehicle.ADAS.LK.Steering": Datapoint(float(st))
    }
    kuksa.set_current_values(updates)
# =================================================================


# LK 관련 유틸 함수, 변수
# ==================================================================
DEF_TOPIC_CAM = "carla/cam/front"  # 공유 메모리에서 받아올 카메라 데이터

def parse_attachment(att: bytes) -> dict:
    try:
        return json.loads(att.decode("utf-8"))
    except Exception:
        return {}

def bgra_to_bgr(buf: memoryview, w: int, h: int) -> np.ndarray:
    return np.frombuffer(buf, np.uint8).reshape((h, w, 4))[:, :, :3].copy()
# ===================================================================


# ACC 관련 유틸 함수
# ==================================================================

# ==================================================================


# --- Main 문 ---
# ==================================================================
def main():
    # ------------------- Argparse 인자 파싱 ----------------------------
    ap = argparse.ArgumentParser()
    # Zenoh 관련 파라미터
    ap.add_argument("--zenoh_endpoint", default="tcp/127.0.0.1:7447")
    ap.add_argument("--topic_cam", default=DEF_TOPIC_CAM)
    ap.add_argument("--fps", type=int, default=20)

    # KUKSA 관련 파라미터
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)    

    # LK 관련 파라미터
    ap.add_argument("--roi_json", type=str, default="", help="픽셀 좌표 폴리곤 JSON (미지정 시 기본 ROI)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--canny_low", type=int, default=60)
    ap.add_argument("--canny_high", type=int, default=180)
    ap.add_argument("--hough_thresh", type=int, default=40)
    ap.add_argument("--hough_min_len", type=int, default=20)
    ap.add_argument("--hough_max_gap", type=int, default=120)
    ap.add_argument("--lane_mem_ttl", type=int, default=60)
    ap.add_argument("--anchor_speed_kmh", type=float, default=50.0)

    # 시각화 관련 파라미터
    ap.add_argument("--display", type=int, default=1)   # 시각화 on/off
    ap.add_argument("--print", type=int, default=1)     # 로그 on/off
    args = ap.parse_args()
    dt = 1.0 / max(1, args.fps)

    # -------------------- KUKSA 연결 -----------------------
    kuksa = VSSClient(args.kuksa_host, args.kuksa_port)
    kuksa.connect()
    print(f"[decision_LK] Kuksa connected @ {args.kuksa_host}:{args.kuksa_port}")

    # -------------------- LK 메인 루프 ---------------------
    # zenoh에서 카메라 데이터 받아와서 로컬 변수로 등록
    latest = {"bgr": None, "meta": {}, "ts": 0.0}
    no_frame_tick = 0

    def _on_cam(sample):
        try:
            # 1) payload 꺼내기 (버전 호환)
            raw = None
            if hasattr(sample, "payload"):
                raw = bytes(sample.payload)
            elif hasattr(sample, "value") and hasattr(sample.value, "payload"):
                raw = bytes(sample.value.payload)
            if raw is None:
                return  # 알 수 없는 형태

            # 2) attachment 꺼내기 (가능하면)
            meta = None
            att_bytes = None
            if hasattr(sample, "attachment"):
                att_bytes = sample.attachment
            elif hasattr(sample, "value") and hasattr(sample.value, "encoding_info"):
                att_bytes = None

            if att_bytes:
                try:
                    meta = json.loads(att_bytes.decode("utf-8"))
                except Exception:
                    meta = None

            # 3) w/h 결정: 메타가 있으면 거기서, 없으면 인자 기본값 사용
            if meta and "w" in meta and "h" in meta:
                w = int(meta["w"]); h = int(meta["h"])
            else:
                w = int(args.width); h = int(args.height)

            # 4) BGRA → BGR
            if len(raw) != w * h * 4:
                # 메타가 없거나 사이즈 불일치 시, 복구 실패 → 스킵(로그만)
                if args.print:
                    print(f"[WARN] bad payload size: got={len(raw)} expected={w*h*4} (w={w},h={h})")
                return
            arr = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            bgr = arr[:, :, :3].copy()

            latest["bgr"] = bgr
            latest["meta"] = meta or {"w": w, "h": h, "frame": -1}
            latest["ts"] = time.time()
        except Exception as e:
            print("[ERR] cam callback:", repr(e))

    # 구독 받아와서 프레임마다 on_cam 콜백
    sub_cam = z.declare_subscriber(args.topic_cam, _on_cam)

    if args.display:
        cv2.namedWindow("LK_decision", cv2.WINDOW_NORMAL)

    lane_mem = {"left": None, "right": None, "t_left": None, "t_right": None, "center_x": None}

    print("[INFO] decision_LK.py . Press q to quit.")

    # -------------------- LK 진행 및 st 값 계산, 산출 ---------------------
    try:
        next_t = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            try:
                if latest["bgr"] is None:
                    no_frame_tick += 1
                    # 1초마다 프레임 미수신 경고 (fps 기준)
                    if args.print and (no_frame_tick % max(1, args.fps) == 0):
                        print("[WAIT] no camera frame yet... "
                              f"topic='{args.topic_cam}', endpoint='{args.zenoh_endpoint}'")
                    # GUI가 켜진 상태에서라도 프레임 없으면 빈 화면 → 그냥 루프 지속
                    if args.display:
                        # 빈 프레임 대신 검은 화면 유지
                        cv2.waitKey(1)
                    continue

                no_frame_tick = 0
                bgr = latest["bgr"]
                h, w = bgr.shape[:2]
                frame_id = int(latest["meta"].get("frame", 0))

                # 차선 검출 (ROI: None → LK_algo 기본 ROI)
                lanes = detect_lanes_and_center(
                    bgr,
                    roi_vertices=None,  # 외부 ROI 쓰려면 여기 전달
                    canny_low=args.canny_low, canny_high=args.canny_high,
                    hough_thresh=args.hough_thresh,
                    hough_min_len=args.hough_min_len,
                    hough_max_gap=args.hough_max_gap,
                )

                used_left, used_right, center_line, _center_x = fuse_lanes_with_memory(
                    lanes, frame_id, lane_mem, ttl_frames=args.lane_mem_ttl
                )

                ##################수정 필요###################
                y_anchor = int(lookahead_ratio(args.anchor_speed_kmh) * h)
                ##############################################
                x_cam_mid = w // 2

                x_lane_mid = lane_mem.get("center_x")
                if used_left and used_right:
                    x_lane_mid, _, _ = lane_mid_x(used_left, used_right, y_anchor)
                    lane_mem["center_x"] = x_lane_mid

                st = 0.0
                if x_lane_mid is not None:
                    offset_px = (x_lane_mid - x_cam_mid)   # +: 우측으로 치우침
                    v_kmh = 0.0
                    kp, st_clip = gains_for_speed(v_kmh / 3.6)  # 함수는 m/s 입력, 여기선 0
                    st = float(max(-st_clip, min(st_clip, kp * offset_px)))  # 좌:-, 우:+ 방향 유지
                else:
                    st = 0.0  # 차선 미검출 시 0으로

                # KUKSA 이용하여 steering Publish
                publish_steer(kuksa, st)             

                ############## 변수 구독 필요
                v_kmh = 0
                thr = 0
                brk = 0
                # st = 0
                ###############
                
                # 로그 표시
                if args.print:
                    print(f"v={v_kmh:5.1f} km/h  thr={thr:.2f} brk={brk:.2f} str={st:+.2f}")

                # 시각화
                if args.display:
                    vis = bgr.copy()
                    rv = lanes.get("roi_vertices")
                    if rv is not None:
                        overlay = vis.copy()
                        cv2.fillPoly(overlay, rv, (0, 0, 255))
                        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0.0)
                        # cv2.polylines(vis, rv, True, (0, 0, 180), 2, cv2.LINE_AA)

                    for x1, y1, x2, y2 in lanes.get("line_segs", []):
                        cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)

                    if used_left:
                        cv2.line(vis, (used_left[0], used_left[1]), (used_left[2], used_left[3]),
                                 (0, 0, 255), 5, cv2.LINE_AA)
                    if used_right:
                        cv2.line(vis, (used_right[0], used_right[1]), (used_right[2], used_right[3]),
                                 (0, 0, 255), 5, cv2.LINE_AA)

                    cv2.line(vis, (x_cam_mid, 0), (x_cam_mid, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
                    if x_lane_mid is not None:
                        cv2.line(vis, (x_lane_mid, 0), (x_lane_mid, h - 1), (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(vis, (0, y_anchor), (w - 1, y_anchor), (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.putText(vis, f"v={v_kmh:5.1f} km/h  thr={thr:.2f} brk={brk:.2f} str={st:+.2f}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

                    cv2.imshow("LK_decision", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break

            except Exception as e:
                # 루프 내부 예외를 보여주고 계속/종료 선택
                print("[ERR] loop:", repr(e))
                break

    finally:
        kuksa.close()
        try: sub_cam.undeclare()
        except Exception as e: print("[WARN] sub_cam.undeclare:", e)
        try: z.close()
        except Exception as e: print("[WARN] zenoh.close:", e)
        if args.display:
            try: cv2.destroyAllWindows()
            except Exception as e: print("[WARN] destroyAllWindows:", e)
        print("[INFO] decision.py stopped.")
# ==================================================================


if __name__ == "__main__":
    main()
