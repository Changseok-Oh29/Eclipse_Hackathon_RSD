#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# statesub_ttc_hysteresis.py — KUKSA 구독: 히스테리시스 기반 상태 + 추가 지연(Δ) 출력 (kuksa-client 0.4.0)

import os, sys, argparse, time
from kuksa_client.grpc import VSSClient

DEFAULT_HOST = os.environ.get("KUKSA_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("KUKSA_PORT", "55555"))

# 상태별 추가 지연 Δ (seconds)
DELTA = {
    "dry":  {"acc_headway_s": 0.0, "aeb_ttc_s": 0.0},
    "wet":  {"acc_headway_s": 1.0, "aeb_ttc_s": 0.8},
    "snow": {"acc_headway_s": 2.5, "aeb_ttc_s": 2.3},
    "icy":  {"acc_headway_s": 4.0, "aeb_ttc_s": 4.0},
}

# 라벨 정규화 + 위험도
CANON = {"dry":"dry","wet":"wet","snow":"snow","snowy":"snow","ice":"icy","icy":"icy","unknown":"unknown"}
PRIO  = {"dry":0,"wet":1,"snow":2,"icy":3,"unknown":-1}

# 히스테리시스 파라미터
MIN_HOLD       = 0.60   # 전환 후 최소 유지(sec)
UNKNOWN_GRACE  = 1.00   # unknown 잠깐은 무시(sec)
DWELL_IN = {            # 더 위험하게 갈 때 필요한 최소 지속시간
    ("dry","wet"):0.15, ("wet","snow"):0.20, ("snow","icy"):0.20,
    ("dry","snow"):0.25, ("wet","icy"):0.25, ("dry","icy"):0.30,
}
DWELL_OUT = {           # 더 안전하게 돌아갈 때 필요한 최소 지속시간
    ("wet","dry"):0.60, ("snow","wet"):0.80, ("icy","snow"):1.00,
    ("snow","dry"):1.00, ("icy","wet"):1.20, ("icy","dry"):1.50,
}

def canon_state(s):
    if s is None: return "unknown"
    s = str(s).strip().lower()
    return CANON.get(s, "unknown")

class HysteresisState:
    def __init__(self, init_state="dry"):
        self.stable = init_state
        self.stable_since = time.time()
        self.candidate = None
        self.candidate_since = None

    def _needed_dwell(self, cur, cand):
        if cur == cand: return 0.0
        if PRIO.get(cand,-1) > PRIO.get(cur,-1):
            return DWELL_IN.get((cur,cand), 0.20)
        else:
            return DWELL_OUT.get((cur,cand), 0.80)

    def feed(self, raw_state: str) -> str:
        now = time.time()
        s = canon_state(raw_state)

        # unknown 짧게는 무시
        if s == "unknown" and (now - self.stable_since) < UNKNOWN_GRACE:
            return self.stable
        if s == "unknown":
            s = "dry"  # grace 초과 시 폴백

        # 전환 후 최소 유지시간
        if (now - self.stable_since) < MIN_HOLD:
            # 유지시간 동안에는 후보 초기화
            self.candidate, self.candidate_since = None, None
            return self.stable

        # 같은 상태면 후보 초기화 후 유지
        if s == self.stable:
            self.candidate, self.candidate_since = None, None
            return self.stable

        # 다른 상태면 후보 관리
        if self.candidate != s:
            self.candidate, self.candidate_since = s, now
            return self.stable

        # 동일 후보가 dwell 필요시간을 채웠는지 확인
        need = self._needed_dwell(self.stable, self.candidate)
        have = now - (self.candidate_since or now)
        if have >= need:
            self.stable = self.candidate
            self.stable_since = now
            self.candidate, self.candidate_since = None, None
        return self.stable

def main():
    ap = argparse.ArgumentParser(description="KUKSA 구독: 히스테리시스 기반 상태 + 추가 지연(Δ) 출력")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--path", default="Vehicle.Private.StateFused.State",
                    help="구독할 VSS 경로 (기본: 퓨전 결과). 예: Vehicle.Private.Road.State")
    # 튜닝 옵션
    ap.add_argument("--hold", type=float, default=MIN_HOLD, help="전환 후 최소 유지(sec)")
    ap.add_argument("--unknown-grace", type=float, default=UNKNOWN_GRACE, help="unknown 무시(sec)")
    args = ap.parse_args()

    cli = VSSClient(args.host, args.port); cli.connect()
    sub = cli.subscribe_current_values([args.path])

    # 인자로 받은 값 적용
    hyst = HysteresisState(init_state="dry")
    hyst.MIN_HOLD = float(args.hold)
    hyst.UNKNOWN_GRACE = float(args.unknown_grace)

    try:
        for update in sub:  # dict[path -> Datapoint]
            dp = update.get(args.path)
            if dp is None or dp.value is None:
                continue
            stable = hyst.feed(dp.value)
            d = DELTA.get(stable, DELTA["dry"])
            print("state: "+f"{stable}  ACCΔ=+{d['acc_headway_s']:.1f}s  AEBΔ=+{d['aeb_ttc_s']:.1f}s")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        try: sub.unsubscribe()
        except Exception: pass
        try: cli.close()
        except Exception: pass

if __name__ == "__main__":
    main()
