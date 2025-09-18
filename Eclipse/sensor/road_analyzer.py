# -*- coding: utf-8 -*-
# zenoh RAW 프레임을 '무복사 뷰'로 받아 분석 (POSIX SHM 불필요)
import os, time, json
import numpy as np
import cv2
import zenoh

# ====== 기존 파라미터 유지 ======
EMA_A   = float(os.environ.get("EMA_A", "0.3"))
SAVE_DEBUG = os.environ.get("SAVE_DEBUG", "1") == "1"
IN_KEY  = os.environ.get("IN_KEY", "carla/cam/front")   # ← RAW 프레임 토픽

ROI_Y0=float(os.environ.get("ROI_Y0","0.5")); ROI_XL=float(os.environ.get("ROI_XL","0.2")); ROI_XR=float(os.environ.get("ROI_XR","0.8"))
TH_SRI_WET=float(os.environ.get("TH_SRI_WET","0.06")); TH_LV_WET=float(os.environ.get("TH_LV_WET","50"))
TH_SRI_ICY=float(os.environ.get("TH_SRI_ICY","0.10")); TH_S_MEAN_I=float(os.environ.get("TH_S_MEAN_I","0.25")); TH_V_MEAN_I=float(os.environ.get("TH_V_MEAN_I","0.70"))
TH_DR_SNOW=float(os.environ.get("TH_DR_SNOW","0.35")); TH_S_MEAN_S=float(os.environ.get("TH_S_MEAN_S","0.20"))

# ====== 기존 함수들(수정 없음) ======
def compute_metrics(bgr, roi_box):
    x0,y0,x1,y1 = roi_box
    roi = bgr[y0:y1, x0:x1]                      # view
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

class EMA:
    def __init__(self,a): self.a=a; self.v={}
    def update(self,d):
        out={}; 
        for k,v in d.items():
            p=self.v.get(k); self.v[k]=v if p is None else (1-self.a)*p+self.a*v
            out[k]=self.v[k]
        return out
ema=EMA(EMA_A)

def classify(m):
    if m['SRI']>TH_SRI_ICY and m['S_mean']<TH_S_MEAN_I and m['V_mean']>TH_V_MEAN_I: st='icy'
    elif m['SRI']>TH_SRI_WET and m['LV']<TH_LV_WET: st='wet'
    elif m['DR']>TH_DR_SNOW and m['S_mean']<TH_S_MEAN_S: st='snow'
    else: st='dry'
    import numpy as np
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

# ====== 호환 유틸 ======
def _to_bytes(x):
    if x is None: return b""
    if isinstance(x, (bytes, bytearray, memoryview)): return bytes(x)
    tb = getattr(x, "to_bytes", None)
    if callable(tb): return tb()
    try: return bytes(x)
    except Exception: return b""

def _attachment_bytes(sample):
    # attachment: 버전에 따라 필드/메서드
    att_attr = getattr(sample, "attachment", None)
    att_obj  = att_attr() if callable(att_attr) else att_attr
    return _to_bytes(att_obj)

def _payload_buffer(sample):
    """
    가능하면 '무복사' memoryview를 반환.
    - 일부 바인딩에선 sample.payload 자체가 버퍼 프로토콜을 구현
    - 아니면 최후에 to_bytes()로 복사
    """
    pay = getattr(sample, "payload", None)
    try:
        return memoryview(pay)  # ★ 최선: 복사 없이 뷰
    except TypeError:
        b = _to_bytes(pay)      # 복사 발생 가능
        return memoryview(b)

def main():
    cfg = zenoh.Config()
    # 라우터 고정 (버전에 따라 insert_json5 필요)
    if hasattr(cfg, "insert_json"):
        cfg.insert_json('connect/endpoints','["tcp/127.0.0.1:7447"]')
    else:
        cfg.insert_json5('connect/endpoints','["tcp/127.0.0.1:7447"]')
    sess = zenoh.open(cfg)
    pub  = sess.declare_publisher("road/state")

    print(f"[Analyzer] subscribing raw: {IN_KEY}")

    def on_raw(sample: zenoh.Sample):
        try:
            # ---- 메타 추출 (attachment: JSON bytes) ----
            meta_b = _attachment_bytes(sample)
            meta = json.loads(meta_b.decode("utf-8")) if meta_b else {}
            w = int(meta.get("w", 0)); h = int(meta.get("h", 0))
            c = int(meta.get("c", 4)); fmt = meta.get("format", "bgra8")
            stride = int(meta.get("stride", w * c))

            if w<=0 or h<=0 or c<3:
                return

            # ---- payload를 무복사 memoryview로 ----
            buf = _payload_buffer(sample)  # memoryview
            # 연속 메모리 전제(stride==w*c) — 퍼블리셔가 w*4로 보냄
            if stride != w * c:
                # 비연속 stride는 고급 처리 필요(여기선 단순히 포기)
                return

            # NumPy '뷰' 만들기 (복사 없이)
            bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, c))
            # BGR 뷰(복사 없이 알파 버림): [:,:,0:3]은 view
            bgr = bgra[:, :, :3]

            # ROI & 분석
            y0=int(h*ROI_Y0); y1=h; x0=int(w*ROI_XL); x1=int(w*ROI_XR)
            m = compute_metrics(bgr, (x0,y0,x1,y1))
            m = ema.update(m)
            st, cf = classify(m)

            msg = {
                "state": st,
                "confidence": round(cf,3),
                "metrics": {k: round(float(v),4) for k,v in m.items()},
                "ts": meta.get("sim_ts"), "frame": meta.get("frame")
            }
            pub.put(json.dumps(msg).encode("utf-8"))

            if SAVE_DEBUG:
                vis = overlay(bgr, m, (x0,y0,x1,y1), st, cf)
                cv2.imwrite("debug_latest.jpg", vis)

            # 가끔 진행 로그
           # if (int(meta.get("frame", 0)) % 30) == 0:
                m_fmt = {k: f"{v:.4f}" for k, v in m.items()}
                print(f"[STATE] {st:<5} conf={cf:.2f} metrics={m_fmt}")
               # print(f"[Analyzer] {st}({cf:.2f}) frame={meta.get('frame')}")

        except Exception as e:
            import traceback; print("callback error"); traceback.print_exc()

    sess.declare_subscriber(IN_KEY, on_raw)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sess.close()

if __name__ == "__main__":
    main()
