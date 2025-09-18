# road_analyzer.py (정리된 최종판)
# -*- coding: utf-8 -*-
import os, json, time
from collections import deque
import numpy as np
import cv2
import zenoh

# ===== 파라미터 =====
EMA_A   = float(os.environ.get("EMA_A", "0.3"))
SAVE_DEBUG = os.environ.get("SAVE_DEBUG", "1") == "1"
IN_KEY  = os.environ.get("IN_KEY", "carla/cam/front")

ROI_Y0=float(os.environ.get("ROI_Y0","0.5"))
ROI_XL=float(os.environ.get("ROI_XL","0.2"))
ROI_XR=float(os.environ.get("ROI_XR","0.8"))
TH_SRI_WET=float(os.environ.get("TH_SRI_WET","0.06")); TH_LV_WET=float(os.environ.get("TH_LV_WET","50"))
TH_SRI_ICY=float(os.environ.get("TH_SRI_ICY","0.10")); TH_S_MEAN_I=float(os.environ.get("TH_S_MEAN_I","0.25")); TH_V_MEAN_I=float(os.environ.get("TH_V_MEAN_I","0.70"))
TH_DR_SNOW=float(os.environ.get("TH_DR_SNOW","0.35")); TH_S_MEAN_S=float(os.environ.get("TH_S_MEAN_S","0.20"))

# ===== 유틸/분석 함수 =====
def compute_metrics(bgr, roi_box):
    x0,y0,x1,y1 = roi_box
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _,S,V = cv2.split(hsv)
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
        out={}
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
    conf=0.5
    if st=='wet':  conf=min(1.0,0.5+max(0,(m['SRI']-TH_SRI_WET)*6)+max(0,(TH_LV_WET-m['LV'])/100))
    elif st=='icy': conf=min(1.0,0.5+max(0,(m['SRI']-TH_SRI_ICY)*5)+max(0,(TH_S_MEAN_I-m['S_mean'])*2)+max(0,(m['V_mean']-TH_V_MEAN_I)*1.5))
    elif st=='snow': conf=min(1.0,0.5+max(0,(m['DR']-TH_DR_SNOW)*1.5)+max(0,(TH_S_MEAN_S-m['S_mean'])*1.5))
    else:           conf=min(1.0,0.5+max(0,(m['LV']-70)/120)+max(0,(TH_SRI_WET-m['SRI'])*4))
    return st, float(np.clip(conf,0.0,1.0))

def overlay(bgr, m, roi_box, st, conf):
    x0,y0,x1,y1=roi_box; vis=bgr.copy()
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2)
    t=f"{st}({conf:.2f}) SRI:{m['SRI']:.3f} LV:{m['LV']:.1f} ED:{m['ED']:.3f} DR:{m['DR']:.3f} S:{m['S_mean']:.3f} V:{m['V_mean']:.3f}"
    cv2.putText(vis,t,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    return vis

def _to_bytes(x):
    if x is None: return b""
    if isinstance(x,(bytes,bytearray,memoryview)): return bytes(x)
    tb=getattr(x,"to_bytes",None)
    if callable(tb): return tb()
    try: return bytes(x)
    except Exception: return b""

def _attachment_bytes(sample):
    att_attr=getattr(sample,"attachment",None)
    att_obj=att_attr() if callable(att_attr) else att_attr
    return _to_bytes(att_obj)

def _payload_buffer(sample):
    pay=getattr(sample,"payload",None)
    try:
        return memoryview(pay)        # 최우선: 무복사 뷰
    except TypeError:
        b=_to_bytes(pay)              # 불가 시 1회 복사
        return memoryview(b)

# ===== 실시간 로깅 상태 =====
_t_prev_arrival=None
_intervals=deque(maxlen=120)

def on_raw_factory(pub):
    """실시간 로그 콜백을 publisher에 바인딩해서 반환"""
    def on_raw(sample: zenoh.Sample):
        global _t_prev_arrival
        t0=time.perf_counter()

        # --- 메타/페이로드 ---
        meta_b=_attachment_bytes(sample)
        meta=json.loads(meta_b.decode("utf-8")) if meta_b else {}
        w=int(meta.get("w",0)); h=int(meta.get("h",0))
        c=int(meta.get("c",4)); stride=int(meta.get("stride", w*c))
        if w<=0 or h<=0 or c<3: return

        buf=_payload_buffer(sample)
        if stride!=w*c: return
        bgra=np.frombuffer(buf,dtype=np.uint8).reshape((h,w,c))
        bgr=bgra[:, :, :3]  # 알파 버림(뷰)

        # --- 입력 FPS ---
        if _t_prev_arrival is not None:
            _intervals.append(t0-_t_prev_arrival)
        _t_prev_arrival=t0
        fps_in=None
        if len(_intervals)>=5:
            avg=sum(_intervals)/len(_intervals); fps_in=(1.0/avg) if avg>0 else None

        # --- 분석 ---
        y0=int(h*ROI_Y0); y1=h; x0=int(w*ROI_XL); x1=int(w*ROI_XR)
        m=compute_metrics(bgr,(x0,y0,x1,y1))
        m=ema.update(m)
        st,cf=classify(m)

        # --- 결과 게시 ---
        out={"state":st,"confidence":round(cf,3),
             "metrics":{k:round(float(v),4) for k,v in m.items()},
             "ts": meta.get("sim_ts"), "frame": meta.get("frame")}
        pub.put(json.dumps(out).encode("utf-8"))

        if SAVE_DEBUG:
            vis=overlay(bgr,m,(x0,y0,x1,y1),st,cf)
            cv2.imwrite("debug_latest.jpg",vis)

        # --- 타이밍 ---
        t1=time.perf_counter()
        proc_ms=(t1-t0)*1000.0
        e2e_ms=None
        # 퍼블리셔가 'pub_ts' (time.time())를 넣어주면 그것으로 E2E 계산
        if meta.get("pub_ts") is not None:
            try: e2e_ms=(time.time()-float(meta["pub_ts"])) * 1000.0
            except Exception: pass

        parts=[f"{st}({cf:.2f})", f"F={meta.get('frame')}", f"proc={proc_ms:.1f}ms"]
        if e2e_ms is not None: parts.append(f"E2E={e2e_ms:.1f}ms")
        if fps_in  is not None: parts.append(f"in={fps_in:.1f}fps")
        print("[Analyzer]", " ".join(parts), flush=True)
    return on_raw

def main():
    # 세션/퍼블리셔 생성
    cfg=zenoh.Config()
    if hasattr(cfg,"insert_json"):
        cfg.insert_json('connect/endpoints','["tcp/127.0.0.1:7447"]')
    else:
        cfg.insert_json5('connect/endpoints','["tcp/127.0.0.1:7447"]')
    sess=zenoh.open(cfg)
    pub=sess.declare_publisher("road/state")

    print(f"[Analyzer] subscribing raw: {IN_KEY}")
    # 위의 실시간 콜백 등록 (중복 정의 없음!)
    sess.declare_subscriber(IN_KEY, on_raw_factory(pub))
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sess.close()

if __name__=="__main__":
    # 콘솔 버퍼링 방지 권장: PYTHONUNBUFFERED=1 python road_analyzer.py
    main()
