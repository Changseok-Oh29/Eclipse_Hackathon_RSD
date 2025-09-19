#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# statesub_ttc.py — KUKSA 구독: 상태 + 추가 지연(Δ) 출력 (kuksa-client 0.4.0)

import os, sys, argparse
from kuksa_client.grpc import VSSClient

DEFAULT_HOST = os.environ.get("KUKSA_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("KUKSA_PORT", "55555"))

# 기본값(참고): base는 계산용이지만, 이 스크립트는 "추가 지연 Δ"만 출력합니다.
BASE = {"acc_headway_s": 2.0, "aeb_ttc_s": 1.5}

# 상태별 "추가 지연(Δ, seconds)" — 요청대로 반영
DELTA = {
    "dry":  {"acc_headway_s": 0.0, "aeb_ttc_s": 0.0},
    "wet":  {"acc_headway_s": 1.0, "aeb_ttc_s": 0.8},
    "snow": {"acc_headway_s": 2.5, "aeb_ttc_s": 2.3},
    "icy":  {"acc_headway_s": 4.0, "aeb_ttc_s": 4.0},
}

CANON = {
    "dry":"dry", "wet":"wet",
    "snow":"snow", "snowy":"snow",
    "ice":"icy", "icy":"icy",
    "unknown":"unknown"
}

def canon_state(s):
    if s is None: return "unknown"
    s = str(s).strip().lower()
    return CANON.get(s, s if s in DELTA else "unknown")

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--path", default="Vehicle.Private.Road.State")
    # 필요하면 퓨전 경로나 다른 경로로 바꿔서 쓰세요:
    # --path Vehicle.Private.StateFused.State
    args, _ = ap.parse_known_args()

    cli = VSSClient(args.host, args.port); cli.connect()
    sub = cli.subscribe_current_values([args.path])

    try:
        for update in sub:  # dict[path -> Datapoint]
            dp = update.get(args.path)
            if dp is None or dp.value is None:
                continue
            st = canon_state(dp.value)
            d  = DELTA.get(st, DELTA["dry"])  # unknown이면 dry와 동일 취급 또는 필요에 맞게 변경
            acc_d = d["acc_headway_s"]
            aeb_d = d["aeb_ttc_s"]
            # 상태 + 추가지연(Δ)만 출력 (예: "wet  ACCΔ=+1.0s  AEBΔ=+0.8s")
            print("state: "+ f"{st}  ACCΔ=+{acc_d:.1f}s  AEBΔ=+{aeb_d:.1f}s")
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
