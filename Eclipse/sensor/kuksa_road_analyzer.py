# -*- coding: utf-8 -*-
# kuksa_road_analyzer.py
#
# 기능 요약
# - Zenoh로 받은 CARLA 카메라 프레임을 가볍게 분석(ROI 기반 SRI/LV/ED 등)
# - 결과를 매 프레임 Kuksa VSS에 반드시 게시 (모델이 느려도 직전 결과 재사용)
# - Road.Ts는 심시간(sim time)만 사용하며 Δt 격자(CAM_DT)로 스냅(quantize)
# - 로그는 한 줄에 여러 [STATE] 블록이 이어지며, LINE_WRAP개마다 줄바꿈
#
# 실행 예:
#   CAM_DT=0.05 FRAME_INTERVAL=1 LINE_WRAP=8 python kuksa_road_analyzer.py

import os, sys, json, time, traceback
import numpy as np
import cv2
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint

# ================== 환경변수 설정 ==================
IN_KEY        = os.environ.get("IN_KEY", "carla/cam/front")  # Zenoh 입력 토픽
KUKSA_HOST    = os.environ.get("KUKSA_HOST", "127.0.0.1")
KUKSA_PORT    = int(os.environ.get("KUKSA_PORT", "55555"))

# 시간/주기: env.py가 fps=20이면 Δt=0.05
CAM_DT        = float(os.environ.get("CAM_DT", "0.05"))  # Δt(s) — slip과 동일해야 함
FRAME_INTERVAL= int(os.environ.get("FRAME_INTERVAL", "1"))  # n프레임마다 새 추론(1=매프레임)
FORCE_QUANT   = os.environ.get("FORCE_QUANTIZE", "1") == "1" # Ts를 Δt 격자로 강제 스냅

# 로깅(요청 포맷)
LINE_WRAP     = int(os.environ.get("LINE_WRAP", "8"))  # 몇 개 찍고 줄바꿈할지
_print_chunk_cnt = 0

# 디버그
SAVE_DEBUG    = os.environ.get("SAVE_DEBUG", "0") == "1"
EMA_A         = float(os.environ.get("EMA_A", "0.3"))

# ROI/임계값 (간단 룰 기반 분류)
ROI_Y0=float(os.environ.get("ROI_Y0","0.5"))
ROI_XL=float(os.environ.get("ROI_XL","0.2"))
ROI_XR=float(os.environ.get("ROI_XR","0.8"))

TH_SRI_WET=float(os.environ.get("TH_SRI_WET","0.06"))
TH_LV_WET=float(os.environ.get("TH_LV_WET","50"))
TH_SRI_ICY=float(os.environ.get("TH_SRI_ICY","0.10"))
TH_S_MEAN_I=float(os.environ.get("TH_S_MEAN_I","0.25"))
TH_V_MEAN_I=float(os.environ.get("TH_V_MEAN_I","0.70"))
TH_DR_SNOW=float(os.environ.get("TH_DR_SNOW","0.35"))
TH_S_MEAN_S=float(os.environ.get("TH_S_MEAN_S","0.20"))

# ================== 유틸 ==================
def _to_bytes(x):
    if x is None: return b""
    if isinstance(x, (bytes, bytearray, memoryview)): return bytes(x)
    try: return bytes(x)
    except Exception: return b""

def _attachment_bytes(sample):
    att_attr = getattr(sample, "attachment", None)
    att_obj  = att_attr() if callable(att_attr) else att_attr
    return _to_bytes(att_obj)

def _payload_buffer(sample):
    pay = getattr(sample, "payload", None)
    try:
        return memoryview(pay)               # 최선: 무복사 뷰
    except TypeError:
        return memoryview(_to_bytes(pay))    # 최후: 복사

class EMA:
    def __init__(self,a): self.a=a; self.v={}
    def update(self,d):
        out={}
        for k,v in d.items():
            p=self.v.get(k); self.v[k]=v if p is None else (1-self.a)*p+self.a*v
            out[k]=self.v[k]
        return out

def compute_metrics(bgr, roi_box):
    x0,y0,x1,y1 = roi_box
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    Sn = S.astype(np.float32)/255.0
    Vn = V.astype(np.float32)/255.0

    # === 반짝임 조건 완화 ===
    highlight = (Vn > 0.70) & (Sn < 0.35)
    SRI = float(highlight.mean())

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    LV = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ED = float((cv2.Canny(gray,80,160) > 0).mean())
    DR = float((gray < 50).mean())
    S_mean = float(Sn.mean())
    V_mean = float(Vn.mean())
    return dict(SRI=SRI, LV=LV, ED=ED, DR=DR, S_mean=S_mean, V_mean=V_mean)

def classify(m):
    import numpy as np
    if m['SRI'] > TH_SRI_ICY and m['S_mean'] < TH_S_MEAN_I and m['V_mean'] > TH_V_MEAN_I:
        st = 'icy'

    # === wet 조건 완화 ===
    elif (m['SRI'] > TH_SRI_WET) or (m['LV'] < TH_LV_WET*1.5 and m['V_mean'] > 0.5):
        st = 'wet'

    elif m['DR'] > TH_DR_SNOW and m['S_mean'] < TH_S_MEAN_S:
        st = 'snow'
    else:
        st = 'dry'

    # --- confidence ---
    if st == 'wet':
        conf = min(1.0, 0.6
                        + max(0,(m['SRI']-TH_SRI_WET)*6)      # SRI 기반
                        + max(0,(TH_LV_WET-m['LV'])/200)      # LV 기반
                        + max(0,(m['V_mean']-0.5)*1.0))       # 밝기 기반
    elif st == 'icy':
        conf = min(1.0, 0.5
                        + max(0,(m['SRI']-TH_SRI_ICY)*5)
                        + max(0,(TH_S_MEAN_I-m['S_mean'])*2)
                        + max(0,(m['V_mean']-TH_V_MEAN_I)*1.5))
    elif st == 'snow':
        conf = min(1.0, 0.5
                        + max(0,(m['DR']-TH_DR_SNOW)*1.5)
                        + max(0,(TH_S_MEAN_S-m['S_mean'])*1.5))
    else:
        conf = min(1.0, 0.5
                        + max(0,(m['LV']-70)/120)
                        + max(0,(TH_SRI_WET-m['SRI'])*4))

    return st, float(np.clip(conf,0.0,1.0))

def overlay(bgr, m, roi_box, st, conf):
    x0,y0,x1,y1=roi_box; vis=bgr.copy()
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2)
    t=f"{st}({conf:.2f}) SRI:{m['SRI']:.3f} LV:{m['LV']:.1f} ED:{m['ED']:.3f} DR:{m['DR']:.3f} S:{m['S_mean']:.3f} V:{m['V_mean']:.3f}"
    cv2.putText(vis,t,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    return vis

# ================== 심시간 처리 ==================
_last_ts = None
_last_state, _last_conf = "unknown", 0.40
_last_sri, _last_ed = None, None

def derive_sim_ts(meta: dict) -> float:
    """
    우선순위: attachment.sim_ts → frame*CAM_DT → 내부 증가
    이후 Δt 격자(CAM_DT)로 스냅(옵션)
    """
    global _last_ts
    sim_ts = None

    # 1) 첨부 sim_ts
    v = meta.get("sim_ts")
    if v is not None:
        try:
            sim_ts = float(v)
        except Exception:
            sim_ts = None

    # 2) frame 기반
    if sim_ts is None:
        fr = meta.get("frame")
        if fr is not None:
            try:
                sim_ts = int(fr) * CAM_DT
            except Exception:
                sim_ts = None

    # 3) 내부 증가
    if sim_ts is None:
        sim_ts = 0.0 if _last_ts is None else (_last_ts + CAM_DT)

    # 4) Δt 격자 스냅
    if FORCE_QUANT and CAM_DT > 0:
        n = int(round(sim_ts / CAM_DT))
        sim_ts = round(n * CAM_DT, 6)

    _last_ts = sim_ts
    return sim_ts

# ================== 메인 ==================
def main():
    # Kuksa
    kc = VSSClient(KUKSA_HOST, KUKSA_PORT); kc.connect()
    print(f"[Analyzer] Kuksa connected @ {KUKSA_HOST}:{KUKSA_PORT}")

    # Zenoh
    zcfg = zenoh.Config()
    try:
        zcfg.insert_json5("connect/endpoints",'["tcp/127.0.0.1:7447"]')
    except AttributeError:
        zcfg.insert_json("connect/endpoints",'["tcp/127.0.0.1:7447"]')
    sess = zenoh.open(zcfg)
    print(f"[Analyzer] subscribing raw: {IN_KEY} (Δt={CAM_DT}s, every {FRAME_INTERVAL} frame(s) infer)")

    ema = EMA(EMA_A)
    frame_cnt = 0

    def publish_cam_vss(sim_ts, st=None, cf=None, sri=None, ed=None):
        """매 프레임 반드시 호출: st/cf가 None이면 직전 값 재사용."""
        global _last_state, _last_conf, _last_sri, _last_ed
        if st is None: st = _last_state
        if cf is None: cf = _last_conf

        updates = {
            "Vehicle.Private.Road.State":      Datapoint(st),
            "Vehicle.Private.Road.Confidence": Datapoint(float(cf)),
            "Vehicle.Private.Road.Ts":         Datapoint(float(sim_ts)),
        }
        if sri is not None:
            updates["Vehicle.Private.Road.Metrics.SRI"] = Datapoint(float(sri))
        if ed is not None:
            updates["Vehicle.Private.Road.Metrics.ED"]  = Datapoint(float(ed))

        kc.set_current_values(updates)

        # 직전 상태 갱신(로깅/재사용)
        _last_state, _last_conf = st, float(cf)
        if sri is not None: _last_sri = float(sri)
        if ed  is not None: _last_ed  = float(ed)

    def on_raw(sample: zenoh.Sample):
        nonlocal frame_cnt
        global _print_chunk_cnt
        try:
            # ---- 메타 파싱 & 심시간 도출 ----
            meta_b = _attachment_bytes(sample)
            meta = json.loads(meta_b.decode("utf-8")) if meta_b else {}
            w = int(meta.get("w", 0)); h = int(meta.get("h", 0))
            c = int(meta.get("c", 4)); stride = int(meta.get("stride", max(0,w*c)))
            sim_ts = derive_sim_ts(meta)

            # ---- 추론 간헐 실행 여부 ----
            do_infer = (FRAME_INTERVAL <= 1) or ((frame_cnt % FRAME_INTERVAL) == 0)

            # ---- 유효 페이로드 확인 ----
            valid_img = (w>0 and h>0 and c>=3 and stride == w*c)

            st=cf=None
            if do_infer and valid_img:
                buf = _payload_buffer(sample)
                bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, c))
                bgr = bgra[:,:,:3]

                # ROI & 메트릭
                y0=int(h*ROI_Y0); y1=h; x0=int(w*ROI_XL); x1=int(w*ROI_XR)
                m = compute_metrics(bgr, (x0,y0,x1,y1))
                m = ema.update(m)
                st, cf = classify(m)

                publish_cam_vss(sim_ts, st, cf, sri=m["SRI"], ed=m["ED"])

                if SAVE_DEBUG:
                    vis = overlay(bgr, m, (x0,y0,x1,y1), st, cf)
                    cv2.imwrite("debug_latest.jpg", vis)
            else:
                # 새 추론 건너뛰어도 매 프레임 Ts는 게시 (직전 결과 재사용)
                publish_cam_vss(sim_ts)

            # ---- 요청 포맷 로깅 ----
            use_state = (st if do_infer and st is not None else _last_state)
            use_conf  = (cf if do_infer and cf is not None else _last_conf)
            use_sri   = (_last_sri if _last_sri is not None else 0.0)
            use_ed    = (_last_ed  if _last_ed  is not None else 0.0)

            print(
                f"[STATE] {use_state} conf={use_conf:.2f} "
                f"SRI={use_sri:.3f} ED={use_ed:.3f} ts={sim_ts:.3f} "
            )

            frame_cnt += 1

        except Exception:
            sys.stdout.write("\n")
            sys.stdout.flush()
            print("[Analyzer] callback error:")
            traceback.print_exc()

    sess.declare_subscriber(IN_KEY, on_raw)

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try: kc.disconnect()
        except: pass
        sess.close()

if __name__ == "__main__":
    main()

