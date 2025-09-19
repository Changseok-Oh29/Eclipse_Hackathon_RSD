# -*- coding: utf-8 -*-
# kuksa_road_analyzer.py
# - zenoh로 들어오는 RAW 프레임(attachment에 w,h,c,stride 포함)을 '무복사 뷰'로 분석
# - 결과를 Kuksa Databroker(VSS)에 set_current_values로 게시
# - 필요 시 기존 zenoh JSON도 함께 발행 가능(OUT_ZENOH=1)

import os, time, json, traceback
import numpy as np
import cv2
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint

# ========= 설정 =========
IN_KEY   = os.environ.get("IN_KEY", "carla/cam/front")  # RAW 프레임 토픽
OUT_ZENOH = os.environ.get("OUT_ZENOH", "0") == "1"     # 제노 JSON 동시 발행 여부 (기본 꺼짐)

KUKSA_HOST = os.environ.get("KUKSA_HOST", "127.0.0.1")
KUKSA_PORT = int(os.environ.get("KUKSA_PORT", "55555"))

SAVE_DEBUG = os.environ.get("SAVE_DEBUG", "1") == "1"
EMA_A   = float(os.environ.get("EMA_A", "0.3"))

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

# ========= 유틸 =========
def _to_bytes(x):
    if x is None: return b""
    if isinstance(x, (bytes, bytearray, memoryview)): return bytes(x)
    tb = getattr(x, "to_bytes", None)
    if callable(tb): return tb()
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
    Sn = S.astype(np.float32)/255.0; Vn = V.astype(np.float32)/255.0
    highlight = (Vn>0.85)&(Sn<0.20)
    SRI=float(highlight.mean())
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    LV=float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ED=float((cv2.Canny(gray,80,160)>0).mean())
    DR=float((gray<50).mean())
    S_mean=float(Sn.mean()); V_mean=float(Vn.mean())
    return dict(SRI=SRI,LV=LV,ED=ED,DR=DR,S_mean=S_mean,V_mean=V_mean)

def classify(m):
    import numpy as np
    if m['SRI']>TH_SRI_ICY and m['S_mean']<TH_S_MEAN_I and m['V_mean']>TH_V_MEAN_I: st='icy'
    elif m['SRI']>TH_SRI_WET and m['LV']<TH_LV_WET: st='wet'
    elif m['DR']>TH_DR_SNOW and m['S_mean']<TH_S_MEAN_S: st='snow'
    else: st='dry'
    conf=0.5
    if st=='wet': conf=min(1.0,0.5+max(0,(m['SRI']-TH_SRI_WET)*6)+max(0,(TH_LV_WET-m['LV'])/100))
    elif st=='icy': conf=min(1.0,0.5+max(0,(m['SRI']-TH_SRI_ICY)*5)+max(0,(TH_S_MEAN_I-m['S_mean'])*2)+max(0,(m['V_mean']-TH_V_MEAN_I)*1.5))
    elif st=='snow': conf=min(1.0,0.5+max(0,(m['DR']-TH_DR_SNOW)*1.5)+max(0,(TH_S_MEAN_S-m['S_mean'])*1.5))
    else: conf=min(1.0,0.5+max(0,(m['LV']-70)/120)+max(0,(TH_SRI_WET-m['SRI'])*4))
    return st, float(np.clip(conf,0.0,1.0))

def overlay(bgr, m, roi_box, st, conf):
    x0,y0,x1,y1=roi_box; vis=bgr.copy()
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2)
    t=f"{st}({conf:.2f}) SRI:{m['SRI']:.3f} LV:{m['LV']:.1f} ED:{m['ED']:.3f} DR:{m['DR']:.3f} S:{m['S_mean']:.3f} V:{m['V_mean']:.3f}"
    cv2.putText(vis,t,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    return vis

def main():
    # Kuksa 연결
    kc = VSSClient(KUKSA_HOST, KUKSA_PORT); kc.connect()
    print(f"[Analyzer] Kuksa connected @ {KUKSA_HOST}:{KUKSA_PORT}")

    # zenoh 연결
    zcfg = zenoh.Config()
    if hasattr(zcfg, "insert_json5"):
        zcfg.insert_json5("connect/endpoints",'["tcp/127.0.0.1:7447"]')
    else:
        zcfg.insert_json("connect/endpoints",'["tcp/127.0.0.1:7447"]')
    sess = zenoh.open(zcfg)
    pub  = sess.declare_publisher("road/state") if OUT_ZENOH else None

    ema = EMA(EMA_A)
    print(f"[Analyzer] subscribing raw: {IN_KEY}")

    def on_raw(sample: zenoh.Sample):
        try:
            meta_b = _attachment_bytes(sample)
            meta = json.loads(meta_b.decode("utf-8")) if meta_b else {}
            w = int(meta.get("w", 0)); h = int(meta.get("h", 0))
            c = int(meta.get("c", 4)); stride = int(meta.get("stride", w*c))
            if w<=0 or h<=0 or c<3 or stride != w*c:
                return

            buf = _payload_buffer(sample)
            bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, c))
            bgr = bgra[:,:,:3]

            y0=int(h*ROI_Y0); y1=h; x0=int(w*ROI_XL); x1=int(w*ROI_XR)
            m = compute_metrics(bgr, (x0,y0,x1,y1))
            m = ema.update(m)
            st, cf = classify(m)

            sim_ts = meta.get("sim_ts")
            now_ts = time.time()
            ts_val = float(sim_ts) if sim_ts is not None else float(now_ts)

            # SRI_rel (퓨저 힌트)
            sri_rel = max(0.0, m["SRI"] - TH_SRI_WET)

            # Kuksa publish (v2 스타일: set_current_values)
            updates = {
                "Vehicle.Private.Road.State":            Datapoint(st),
                "Vehicle.Private.Road.Confidence":       Datapoint(float(cf)),
                "Vehicle.Private.Road.Ts":               Datapoint(ts_val),
                "Vehicle.Private.Road.Metrics.SRI":      Datapoint(float(m["SRI"])),
                "Vehicle.Private.Road.Metrics.SRI_rel":  Datapoint(float(sri_rel)),
                "Vehicle.Private.Road.Metrics.ED":       Datapoint(float(m["ED"])),
            }
            kc.set_current_values(updates)

            # (옵션) 제노 JSON 동시 발행
            if OUT_ZENOH and pub is not None:
                msg = {
                    "state": st,
                    "confidence": round(float(cf),3),
                    "metrics": {
                        "SRI": round(float(m["SRI"]),4),
                        "SRI_rel": round(float(sri_rel),4),
                        "ED": round(float(m["ED"]),4)
                    },
                    "ts": ts_val,
                    "frame": meta.get("frame")
                }
                pub.put(json.dumps(msg).encode("utf-8"))

            if SAVE_DEBUG:
                vis = overlay(bgr, m, (x0,y0,x1,y1), st, cf)
                cv2.imwrite("debug_latest.jpg", vis)

            # 진행 로그 (과하지 않게)
            print(f"[STATE] {st:<5} conf={cf:.2f} SRI={m['SRI']:.3f} ED={m['ED']:.3f} ts={ts_val:.3f}")

        except Exception:
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
