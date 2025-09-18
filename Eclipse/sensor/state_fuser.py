# -*- coding: utf-8 -*-
import json, time, os
import zenoh
from collections import deque

KEY_CAM  = os.environ.get("KEY_CAM",  "road/state")
KEY_SLIP = os.environ.get("KEY_SLIP", "slip/state")
KEY_OUT  = os.environ.get("KEY_OUT",  "road/state_fused")

LATENCY      = float(os.environ.get("FUSE_LATENCY", "0.50"))  # 매칭 허용창(초) ← 넉넉히
PRINT_EVERY  = float(os.environ.get("PRINT_EVERY",  "1.0"))   # 상태 로그 주기(초)
FORWARD_ONLY = os.environ.get("FORWARD_ONLY", "0") == "1"     # 디버깅용: cam만 들어와도 바로 내보냄

buf_cam  = deque(maxlen=600)
buf_slip = deque(maxlen=600)

def as_bytes(p):
    if isinstance(p, (bytes, bytearray)):
        return bytes(p)
    if hasattr(p, "to_bytes"):
        return p.to_bytes()
    if hasattr(p, "to_string"):
        return p.to_string().encode("utf-8", "ignore")
    return bytes(p)

def best_match(ts, buf, tol=LATENCY):
    best, bdt = None, tol + 1.0
    for x in buf:
        xts = float(x.get("ts", 0.0))
        dt  = abs(xts - ts)
        if dt < bdt and dt <= tol:
            best, bdt = x, dt
    return best, (bdt if best else None)

def score_from_state(s):
    labels = ["dry","wet","icy","snow","unknown"]
    d = {k:0.0 for k in labels}
    d[s] = d.get(s,0.0) + 1.0
    return d

def fuse(cam, slip):
    # 기본 가중치
    w_cam, w_slip = 0.6, 0.4
    if slip:
        q = float(slip.get("quality",0.0))
        if q >= 0.9: w_slip, w_cam = 0.7, 0.3

    sc = score_from_state(cam.get("state","unknown")) if cam else {}
    ss = score_from_state(slip.get("state","unknown")) if slip else {}

    # 카메라 힌트가 반사↑/엣지↓면 저마찰 가중
    m = cam.get("metrics",{}) if cam else {}
    sri_rel = float(m.get("SRI_rel", 0.0))
    ed      = float(m.get("ED", 0.0))
    if sri_rel > 0.01 and ed < 0.05:
        ss["icy"] = ss.get("icy",0.0) + 0.2
        ss["wet"] = ss.get("wet",0.0) + 0.1

    labels = set(sc.keys()) | set(ss.keys())
    total = {k: w_cam*sc.get(k,0.0) + w_slip*ss.get(k,0.0) for k in labels}
    if not total:
        return "unknown", 0.0, {}
    state = max(total.items(), key=lambda kv: kv[1])[0]
    conf  = min(0.99, max(total.values()))
    return state, round(conf,3), total

def main():
    # (권장) 모든 프로세스에서 동일하게 설정
    # os.environ.setdefault("ZENOH_CONNECT", "tcp/127.0.0.1:7447")

    s = zenoh.open(zenoh.Config())
    pub = s.declare_publisher(KEY_OUT)
    print(f"[FUSER] sub: {KEY_CAM},{KEY_SLIP}  pub: {KEY_OUT}  (LAT={LATENCY}s)")

    last_log = 0.0

    def maybe_log():
        nonlocal last_log
        now = time.time()
        if now - last_log >= PRINT_EVERY:
            last_log = now
            print(f"[FUSER] hb cam={len(buf_cam)} slip={len(buf_slip)}")

    def handle_and_publish(trigger_src, obj):
        ts = float(obj.get("ts", time.time()))
        if trigger_src == "cam":
            slip, dt = best_match(ts, buf_slip)
            cam = obj
        else:
            cam, dt = best_match(ts, buf_cam)
            slip = obj

        # 매칭 실패 시에도 cam만 포워드(디버그/옵션)
        if not slip and trigger_src == "cam" and FORWARD_ONLY:
            fused = {
                "state": cam.get("state","unknown"),
                "confidence": cam.get("confidence",0.0),
                "ts": time.time(), "trigger": "cam-only",
                "match_dt": None, "cam": cam, "slip": None, "score": {}
            }
            pub.put(json.dumps(fused).encode("utf-8"))
            print(f"[FUSER] out (cam-only) state={fused['state']} conf={fused['confidence']}")
            return

        state, conf, score = fuse(cam, slip)
        fused = {
            "state": state, "confidence": conf, "ts": time.time(),
            "trigger": trigger_src, "match_dt": (round(dt,3) if dt is not None else None),
            "cam": cam, "slip": slip, "score": score
        }
        pub.put(json.dumps(fused).encode("utf-8"))
        print(f"[FUSER] out state={state:>5} conf={conf:.2f} "
              f"match_dt={fused['match_dt']} (cam={cam is not None}, slip={slip is not None})")

    # ★ 구독 객체를 변수에 보관(참조 유지) → GC로 사라지는 문제 방지
    sub_cam  = s.declare_subscriber(KEY_CAM,  lambda smp: (
        buf_cam.append(json.loads(as_bytes(smp.payload).decode("utf-8","ignore"))),
        handle_and_publish("cam",  buf_cam[-1]),
        maybe_log()
    ))
    sub_slip = s.declare_subscriber(KEY_SLIP, lambda smp: (
        buf_slip.append(json.loads(as_bytes(smp.payload).decode("utf-8","ignore"))),
        handle_and_publish("slip", buf_slip[-1]),
        maybe_log()
    ))

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        s.close()

if __name__ == "__main__":
    main()
