#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuksa State Fuser — Kuksa I/O, zenoh-style logic & logs

요구사항 반영:
- 입출력은 **Kuksa Databroker** 유지 (grpc VSSClient)
- 내부 로직/로그는 첨부된 zenoh fuser(state_fuser.py)의 스타일을 따름:
  - 슬라이딩 버퍼, `best_match()`로 LATENCY 창 내 최근치 매칭
  - 점수 기반 `fuse()` (카메라 SRI_rel/ED로 slip 가중치 bias)
  - 트리거 중심 처리(`handle_and_publish('cam'|'slip')`)와 하트비트
  - 옵션 `FORWARD_ONLY=1` 이면 카메라만 있어도 퍼블리시
- 둘 중 하나가 꺼져도 동작(포워드 또는 단독 소스 융합)

ENV
---
KUKSA_HOST=127.0.0.1
KUKSA_PORT=55555
POLL_DT=0.05                 # 폴링 주기(벽시계)
FUSE_LATENCY=0.50            # 매칭 허용 창(초)
PRINT_EVERY=1.0              # 하트비트 주기(초)
FORWARD_ONLY=1               # 1이면 cam만 들어와도 바로 fused 출력
VERBOSE=1                    # 0/1/2

VSS Paths (필요 최소):
- Camera:  Vehicle.Private.Road.{State,Confidence,Ts,Metrics.SRI_rel,Metrics.ED}
- Slip:    Vehicle.Private.Slip.{State,Quality,Ts}
- Fused→Kuksa: Vehicle.Private.StateFused.{State,Confidence,Ts,Metrics.W_cam,Metrics.W_slip,Metrics.LatencyMs}
"""
import os, time, threading, collections, traceback
from typing import Optional, Dict, Any, Tuple
from kuksa_client.grpc import VSSClient, Datapoint

# -----------------------------
# ENV
# -----------------------------
KUKSA_HOST    = os.environ.get("KUKSA_HOST", "127.0.0.1")
KUKSA_PORT    = int(os.environ.get("KUKSA_PORT", "55555"))
POLL_DT       = float(os.environ.get("POLL_DT", "0.05"))
LATENCY       = float(os.environ.get("FUSE_LATENCY", "0.50"))
PRINT_EVERY   = float(os.environ.get("PRINT_EVERY", "1.0"))
FORWARD_ONLY  = os.environ.get("FORWARD_ONLY", "1") == "1"
VERBOSE       = int(os.environ.get("VERBOSE", "1"))

# Buffers
buf_cam  = collections.deque(maxlen=600)
buf_slip = collections.deque(maxlen=600)

# VSS Paths
CAM_PATHS = [
    "Vehicle.Private.Road.State",
    "Vehicle.Private.Road.Confidence",
    "Vehicle.Private.Road.Ts",
    "Vehicle.Private.Road.Metrics.SRI_rel",
    "Vehicle.Private.Road.Metrics.ED",
]
SLIP_PATHS = [
    "Vehicle.Private.Slip.State",
    "Vehicle.Private.Slip.Quality",
    "Vehicle.Private.Slip.Ts",
]
STATEFUSED_PATHS = {
    "State":      "Vehicle.Private.StateFused.State",
    "Confidence": "Vehicle.Private.StateFused.Confidence",
    "Ts":         "Vehicle.Private.StateFused.Ts",
    "W_cam":      "Vehicle.Private.StateFused.Metrics.W_cam",
    "W_slip":     "Vehicle.Private.StateFused.Metrics.W_slip",
    "LatencyMs":  "Vehicle.Private.StateFused.Metrics.LatencyMs",
}

# -----------------------------
# Helpers (zenoh-style)
# -----------------------------
LABELS = ["dry","wet","icy","snow","unknown"]
CANON  = {"icy":"icy","ice":"icy","snowy":"snow","snow":"snow","wet":"wet","dry":"dry"}

def canon_state(s: Optional[str]) -> str:
    s = (s or "unknown").strip().lower()
    return CANON.get(s, s if s in LABELS else "unknown")

def score_from_state(s: str) -> Dict[str, float]:
    d = {k:0.0 for k in LABELS}
    d[canon_state(s)] = d.get(canon_state(s), 0.0) + 1.0
    return d

def best_match(ts: float, buf: collections.deque, tol: float = LATENCY) -> Tuple[Optional[Dict[str,Any]], Optional[float]]:
    best, bdt = None, tol + 1.0
    for x in buf:
        xts = float(x.get("ts", 0.0))
        dt  = abs(xts - ts)
        if dt < bdt and dt <= tol:
            best, bdt = x, dt
    return best, (bdt if best else None)

def fuse(cam: Optional[Dict[str,Any]], slip: Optional[Dict[str,Any]]):
    # 기본 가중치
    w_cam, w_slip = 0.6, 0.4
    if slip:
        q = float(slip.get("quality", 0.0))
        if q >= 0.9:
            w_slip, w_cam = 0.7, 0.3
    sc = score_from_state(cam.get("state","unknown")) if cam else {}
    ss = score_from_state(slip.get("state","unknown")) if slip else {}
    # 카메라 메트릭으로 slip 저마찰 bias
    m = cam.get("metrics", {}) if cam else {}
    sri_rel = float(m.get("SRI_rel", 0.0))
    ed      = float(m.get("ED", 0.0))
    if sri_rel > 0.01 and ed < 0.05:
        ss["icy"] = ss.get("icy", 0.0) + 0.2
        ss["wet"] = ss.get("wet", 0.0) + 0.1
    labels = set(sc.keys()) | set(ss.keys())
    total = {k: w_cam*sc.get(k,0.0) + w_slip*ss.get(k,0.0) for k in labels}
    if not total:
        return "unknown", 0.0, w_cam, w_slip, {}
    state = max(total.items(), key=lambda kv: kv[1])[0]
    conf  = min(0.99, max(total.values()))
    return state, round(conf,3), w_cam, w_slip, total

# -----------------------------
# Pollers (Kuksa)
# -----------------------------
class Poller:
    def __init__(self, paths, buf, name="CAM"):
        self.paths = paths
        self.buf   = buf
        self.name  = name
        self.last_ts = None
        self.cli = VSSClient(KUKSA_HOST, KUKSA_PORT); self.cli.connect()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        while True:
            try:
                vals = self.cli.get_current_values(self.paths)
                ts_v = vals.get(self.paths[2])  # Ts
                ts   = float(ts_v.value) if ts_v is not None and ts_v.value is not None else None
                if ts is None:
                    time.sleep(POLL_DT); continue  # ts 없으면 스킵
                if self.name == "CAM":
                    obj = {
                        "state": vals.get(self.paths[0]).value if vals.get(self.paths[0]) else "unknown",
                        "confidence": float(vals.get(self.paths[1]).value) if vals.get(self.paths[1]) else 0.0,
                        "ts": ts,
                        "metrics": {
                            "SRI_rel": float(vals.get(self.paths[3]).value) if vals.get(self.paths[3]) else 0.0,
                            "ED":      float(vals.get(self.paths[4]).value) if vals.get(self.paths[4]) else 0.0,
                        }
                    }
                else:
                    obj = {
                        "state": vals.get(self.paths[0]).value if vals.get(self.paths[0]) else "unknown",
                        "quality": float(vals.get(self.paths[1]).value) if vals.get(self.paths[1]) else 0.0,
                        "ts": ts,
                    }
                self.buf.append(obj)
                # 새 샘플이면 트리거
                if self.last_ts is None or ts != self.last_ts:
                    self.last_ts = ts
                    handle_and_publish("cam" if self.name=="CAM" else "slip", obj)
                if VERBOSE>=2:
                    if self.name=="CAM":
                        print(f"[CAM] {obj['state']} conf={obj['confidence']:.2f} SRI_rel={obj['metrics']['SRI_rel']:.3f} ED={obj['metrics']['ED']:.3f} ts={obj['ts']:.3f}")
                    else:
                        print(f"[SLIP] {obj['state']} q={obj['quality']:.2f} ts={obj['ts']:.3f}")
            except Exception:
                time.sleep(POLL_DT)
            time.sleep(POLL_DT)

# -----------------------------
# Kuksa Publisher for fused
# -----------------------------
class FusedPublisher:
    def __init__(self):
        self.cli = VSSClient(KUKSA_HOST, KUKSA_PORT); self.cli.connect()
    def publish(self, state: str, conf: float, ts: float, wc: float, ws: float, lat_ms: float):
        try:
            self.cli.set_current_values({
                STATEFUSED_PATHS["State"]:      Datapoint(state),
                STATEFUSED_PATHS["Confidence"]: Datapoint(float(conf)),
                STATEFUSED_PATHS["Ts"]:         Datapoint(float(ts)),
                STATEFUSED_PATHS["W_cam"]:      Datapoint(float(wc)),
                STATEFUSED_PATHS["W_slip"]:     Datapoint(float(ws)),
                STATEFUSED_PATHS["LatencyMs"]:  Datapoint(float(lat_ms)),
            })
        except Exception:
            traceback.print_exc()

PUB = FusedPublisher()
last_hb = 0.0

def maybe_log():
    global last_hb
    now = time.time()
    if now - last_hb >= PRINT_EVERY:
        last_hb = now
        print(f"[FUSER] hb cam={len(buf_cam)} slip={len(buf_slip)}")

# -----------------------------
# Core trigger handler (zenoh-style)
# -----------------------------

def handle_and_publish(trigger_src: str, obj: Dict[str,Any]):
    ts = float(obj.get("ts", time.time()))
    if trigger_src == "cam":
        slip, dt = best_match(ts, buf_slip)
        cam = obj
    else:
        cam, dt = best_match(ts, buf_cam)
        slip = obj

    # 매칭 실패 시 cam-only forward (옵션)
    if (not slip) and trigger_src == "cam" and FORWARD_ONLY:
        state = canon_state(cam.get("state")); conf = float(cam.get("confidence",0.0))
        PUB.publish(state, conf, ts, wc=1.0, ws=0.0, lat_ms=0.0)
        print(f"[FUSER] out (cam-only) state={state} conf={conf:.2f}")
        maybe_log()
        return

    # 융합
    state, conf, wc, ws, score = fuse(cam, slip)
    lat_ms = (dt*1000.0) if dt is not None else 0.0
    # ts 선택: cam 우선, 없으면 slip, 마지막으로 now()
    ts_out = cam.get("ts") if cam else (slip.get("ts") if slip else ts)

    PUB.publish(state, conf, ts_out, wc, ws, lat_ms)
    print(f"[FUSER] out state={state:>5} conf={conf:.2f} match_dt={(None if dt is None else round(dt,3))} "
          f"(cam={cam is not None}, slip={slip is not None})")
    maybe_log()

# -----------------------------
# Main
# -----------------------------

def main():
    print(f"[FUSER] Kuksa-only I/O, zenoh-style logic (LAT={LATENCY:.2f}s, FORWARD_ONLY={int(FORWARD_ONLY)})")
    # Start pollers (Kuksa)
    Poller(CAM_PATHS,  buf_cam,  name="CAM")
    Poller(SLIP_PATHS, buf_slip, name="SLIP")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
