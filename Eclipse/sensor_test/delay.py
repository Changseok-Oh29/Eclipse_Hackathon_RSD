# -*- coding: utf-8 -*-
import os, time, json
import numpy as np
import cv2
import zenoh
from collections import deque, Counter

IN_KEY  = os.environ.get("IN_KEY", "carla/cam/front/jpeg")
OUT_KEY = os.environ.get("OUT_KEY", "road/state")
EMA_A   = float(os.environ.get("EMA_A", "0.3"))
SAVE_DEBUG = os.environ.get("SAVE_DEBUG", "0") == "1"

# ROI 기본값
ROI_Y0 = float(os.environ.get("ROI_Y0", "0.5"))
ROI_XL = float(os.environ.get("ROI_XL", "0.2"))
ROI_XR = float(os.environ.get("ROI_XR", "0.8"))

# 임계값 (환경변수로 조정 가능)
TH_SRI_WET   = float(os.environ.get("TH_SRI_WET",  "0.06"))
TH_LV_WET    = float(os.environ.get("TH_LV_WET",   "70.0"))
TH_SRI_ICY   = float(os.environ.get("TH_SRI_ICY",  "0.16"))
TH_S_MEAN_I  = float(os.environ.get("TH_S_MEAN_I", "0.25"))
TH_V_MEAN_I  = float(os.environ.get("TH_V_MEAN_I", "0.70"))
TH_S_MEAN_S  = float(os.environ.get("TH_S_MEAN_S", "0.20"))

# 추가 임계값
TH_ED_WET     = float(os.environ.get("TH_ED_WET",     "0.06"))
TH_ED_ICY     = float(os.environ.get("TH_ED_ICY",     "0.035"))
TH_ED_SNOW    = float(os.environ.get("TH_ED_SNOW",    "0.040"))
TH_VTAIL_WET  = float(os.environ.get("TH_VTAIL_WET",  "0.040"))
TH_VTAIL_ICY  = float(os.environ.get("TH_VTAIL_ICY",  "0.060"))
TH_BR_SNOW    = float(os.environ.get("TH_BR_SNOW",    "0.40"))
TH_SLOW_SNOW  = float(os.environ.get("TH_SLOW_SNOW",  "0.70"))
TH_DR_ICY_MAX = float(os.environ.get("TH_DR_ICY_MAX", "0.25"))
SMOOTH_K      = int(os.environ.get("SMOOTH_K",        "5"))

def as_bytes(payload):
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes()
    if hasattr(payload, "to_string"):
        return payload.to_string().encode("utf-8", "ignore")
    return bytes(payload)

class EMA:
    def __init__(self, a): self.a=a; self.v={}
    def update(self, d):
        out={}
        for k,v in d.items():
            prev = self.v.get(k)
            self.v[k] = v if prev is None else (1-self.a)*prev + self.a*v
            out[k] = self.v[k]
        return out
ema = EMA(EMA_A)

def _finite(x, default=0.0):
    try:
        xv = float(x)
        return xv if np.isfinite(xv) else default
    except Exception:
        return default

class StateSmoother:
    def __init__(self, k=5):
        self.buf = deque(maxlen=k)
    def update(self, state):
        self.buf.append(state)
        if not self.buf: return state
        return Counter(self.buf).most_common(1)[0][0]
state_smoother = StateSmoother(SMOOTH_K)

def compute_metrics(bgr):
    h, w = bgr.shape[:2]
    y0 = int(h * ROI_Y0); y1 = h
    x0 = int(w * ROI_XL); x1 = int(w * ROI_XR)
    if y0 >= y1: y0 = max(0, h//2)
    if x0 >= x1: x0, x1 = int(w*0.2), int(w*0.8)
    roi = bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    Sn = S.astype(np.float32)/255.0
    Vn = V.astype(np.float32)/255.0

    highlight = (Vn > 0.85) & (Sn < 0.20)
    SRI = float(highlight.mean())

    v95 = float(np.percentile(Vn, 95)) if Vn.size>0 else 1.0
    SRI_rel = float(((Vn > max(0.5, v95 - 0.05)) & (Sn < 0.25)).mean())

    v90 = float(np.percentile(Vn, 90)) if Vn.size>0 else 1.0
    v99 = float(np.percentile(Vn, 99)) if Vn.size>0 else 1.0
    V_tail = float(max(0.0, v99 - v90))
    BR = float((Vn > 0.80).mean())
    S_low = float((Sn < 0.20).mean())

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    LV  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ED  = float((cv2.Canny(gray, 80, 160) > 0).mean())
    DR  = float((gray < 50).mean())
    S_mean = float(Sn.mean())
    V_mean = float(Vn.mean())

    return dict(
        SRI=SRI, LV=LV, ED=ED, DR=DR,
        S_mean=S_mean, V_mean=V_mean, SRI_rel=SRI_rel,
        V_tail=V_tail, BR=BR, S_low=S_low
    ), roi, (x0,y0,x1,y1)

def classify(m):
    for k in list(m.keys()):
        m[k] = _finite(m[k], 0.0)

    # 눈
    if (m['BR'] > TH_BR_SNOW and m['S_low'] > TH_SLOW_SNOW and m['ED'] < TH_ED_SNOW):
        return 'snow', 0.8

    # 얼음
    if ((m['SRI'] > TH_SRI_ICY or m['V_tail'] > TH_VTAIL_ICY) and
        m['S_mean'] < TH_S_MEAN_I and m['V_mean'] > TH_V_MEAN_I and
        m['ED'] < TH_ED_ICY and m['DR'] < TH_DR_ICY_MAX):
        return 'icy', 0.8

    # 건조 (원래 wet 조건이었던 부분 → dry로!)
    if ((m['SRI'] > TH_SRI_WET and m['LV'] < TH_LV_WET) or
        (m['SRI_rel'] > 0.006 and m['ED'] < TH_ED_WET) or
        (m['V_tail'] > TH_VTAIL_WET and m['ED'] < TH_ED_WET)):
        return 'dry', 0.8

    # 기본: 젖음 (반대로 swap)
    return 'wet', 0.8

def overlay_debug(img, m, roi_rect, state, conf):
    x0,y0,x1,y1 = roi_rect
    vis = img.copy()
    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
    text = f"{state} ({conf:.2f}) SRI:{m['SRI']:.3f} SRIr:{m['SRI_rel']:.3f} LV:{m['LV']:.1f} ED:{m['ED']:.3f} DR:{m['DR']:.3f} S:{m['S_mean']:.3f} V:{m['V_mean']:.3f} Vt:{m['V_tail']:.3f}"
    cv2.putText(vis, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    return vis

# ===== 로깅 상태 =====
_last_arrival = None
_intervals = deque(maxlen=120)
_local_frame_counter = 0

def main():
    conf = zenoh.Config()  # ZENOH_CONNECT 사용 가능
    session = zenoh.open(conf)
    pub = session.declare_publisher(OUT_KEY)
    print(f"[ZENOH] in='{IN_KEY}' out='{OUT_KEY}'")

    def _get_attachment_bytes(sample):
        att_attr = getattr(sample, "attachment", None)
        att_obj  = att_attr() if callable(att_attr) else att_attr
        if att_obj is None:
            return b""
        if isinstance(att_obj, (bytes, bytearray, memoryview)):
            return bytes(att_obj)
        try:
            return bytes(att_obj)
        except Exception:
            return b""

    def on_frame(sample):
        global _last_arrival, _local_frame_counter
        t0 = time.perf_counter()

        try:
            # 입력 FPS 계산용 도착 간격
            if _last_arrival is not None:
                _intervals.append(t0 - _last_arrival)
            _last_arrival = t0

            # payload → JPEG 디코드
            data = as_bytes(sample.payload)
            arr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None: return

            # 메타(프레임 번호 등)
            meta = {}
            att_b = _get_attachment_bytes(sample)
            if att_b:
                try: meta = json.loads(att_b.decode("utf-8"))
                except: meta = {}

            metrics, roi, rect = compute_metrics(bgr)
            smoothed = ema.update(metrics)
            state_raw, cf = classify(smoothed)
            state = state_smoother.update(state_raw)



            # 퍼블리시(기존 유지)
            msg = {
                "state": state, "state_raw": state_raw,
                "confidence": round(cf, 3),
                "metrics": {k: round(float(v),4) for k,v in smoothed.items()},
                "ts": time.time()
            }
            pub.put(json.dumps(msg).encode("utf-8"))

            if SAVE_DEBUG:
                vis = overlay_debug(bgr, smoothed, rect, state, cf)
                cv2.imwrite("debug_latest.jpg", vis)

            # 처리시간/프레임/입력 FPS
            _local_frame_counter += 1
            fno = meta.get("frame", _local_frame_counter)

            t1 = time.perf_counter()
            proc_ms = (t1 - t0) * 1000.0

            in_fps_str = ""
            if len(_intervals) >= 5:
                avg = sum(_intervals)/len(_intervals)
                if avg > 0:
                    in_fps_str = f" in={(1.0/avg):.1f}fps"

            e2e_str = ""
            if meta.get("pub_ts") is not None:
                try:
                    e2e_ms = (time.time() - float(meta["pub_ts"])) * 1000.0
                    e2e_str = f" E2E={e2e_ms:.1f}ms"
                except:
                    pass

            print(f"[Analyzer] {state}({cf:.2f}) F={fno} proc={proc_ms:.1f}ms{e2e_str}{in_fps_str}", flush=True)


        except Exception as e:
            print("[Analyzer] callback error:", e)

    sub = session.declare_subscriber(IN_KEY, on_frame)
    print("[Analyzer] running... Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()

if __name__ == "__main__":
    main()
