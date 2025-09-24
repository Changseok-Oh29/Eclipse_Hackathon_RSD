#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
control.py — Passive (Env가 world.tick 전담)
- lead: TM Autopilot ON
- ego: Autopilot OFF, throttle-only로 ≈25 km/h 전진
- steering: KUKSA Vehicle.ADAS.LK.Steering '원시값(raw)'만 그대로 적용 (필터/데드밴드/레이트리밋 없음)
- 이 파일은 world 설정 변경/ world.tick()/ wait_for_tick()을 절대 호출하지 않음
"""

import time
import math
import argparse
import traceback
import carla

# ---------- KUKSA 최소 래퍼 ----------
DatapointType = None
try:
    # 신/구 혼용 대비
    from kuksa_client.grpc import VSSClient as DataBrokerClient
    try:
        from kuksa_client.grpc import Datapoint as _DP  # 일부 버전만 존재
        DatapointType = _DP
    except Exception:
        DatapointType = None
except Exception:
    try:
        from kuksa_client.grpc import DataBrokerClient
        try:
            from kuksa_client.grpc import Datapoint as _DP
            DatapointType = _DP
        except Exception:
            DatapointType = None
    except Exception:
        DataBrokerClient = None
        DatapointType = None


class KUKSA:
    def __init__(self, host="127.0.0.1", port=55555):
        self.cli = None
        if DataBrokerClient is None:
            print("[KUKSA] sdk not available → steering defaults to 0.0")
            return
        try:
            try:
                self.cli = DataBrokerClient(host, port)
            except:
                self.cli = DataBrokerClient(f"{host}:{port}")
            if hasattr(self.cli, "connect"):
                try:
                    self.cli.connect()
                except Exception:
                    pass
            print("[KUKSA] connected")
        except Exception as e:
            print("[KUKSA] connect failed:", e)
            self.cli = None

    @staticmethod
    def _unwrap(v):
        """
        다양한 반환 형태를 안전하게 원시 값으로 변환:
        - Datapoint(value=...) 객체
        - {"value": ...} dict
        - [Datapoint(...)] / [{"value": ...}] / 기타 시퀀스
        - 단일 스칼라(str/int/float)
        """
        # Datapoint
        if DatapointType is not None and isinstance(v, DatapointType):
            return getattr(v, "value", None)

        # dict with "value"
        if isinstance(v, dict) and "value" in v:
            return v["value"]

        # 리스트/튜플 등 시퀀스 → 첫 원소 재귀 처리
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return KUKSA._unwrap(v[0])

        # 스칼라
        return v

    def get(self, key):
        if self.cli is None:
            return None
        try:
            # 구버전 API들
            if hasattr(self.cli, "get"):
                return self._unwrap(self.cli.get(key))
            if hasattr(self.cli, "get_value"):
                return self._unwrap(self.cli.get_value(key))
            if hasattr(self.cli, "read"):
                return self._unwrap(self.cli.read(key))

            # (신규) KUKSA.val 스타일
            if hasattr(self.cli, "get_current_value"):
                return self._unwrap(self.cli.get_current_value(key))
            if hasattr(self.cli, "get_current_values"):
                res = self.cli.get_current_values([key])
                # dict 형태 { key: Datapoint or {"value": ...} } 혹은 시퀀스
                if isinstance(res, dict):
                    return self._unwrap(res.get(key))
                return self._unwrap(res)
        except Exception:
            pass
        return None


# ---------- 유틸 ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _to_float_safe(x):
    """
    여러 형태로 올 수 있는 값을 float로 최대한 안전하게 변환.
    변환 실패 시 None 반환.
    """
    try:
        return float(x)
    except Exception:
        # dict/Datapoint/리스트 등이 여기까지 내려올 수 있으므로
        if isinstance(x, dict) and "value" in x:
            try:
                return float(x["value"])
            except Exception:
                return None
        if isinstance(x, (list, tuple)) and x:
            return _to_float_safe(x[0])
        # Datapoint 타입이지만 위에서 풀리지 않은 케이스(예외적)
        if DatapointType is not None and isinstance(x, DatapointType):
            try:
                return float(getattr(x, "value", None))
            except Exception:
                return None
        return None


def find_actor(world, role="ego", wait_sec=15.0):
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        for v in world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name") == role:
                print(f"[INFO] {role} found: id={v.id}")
                return v
        time.sleep(0.05)
    print(f"[ERR] {role} not found (role_name='{role}')")
    return None


def speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla_host", default="127.0.0.1")
    ap.add_argument("--carla_port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)   # lead용 TM
    ap.add_argument("--ego_role", default="ego")
    ap.add_argument("--lead_role", default="lead")
    ap.add_argument("--hz", type=float, default=20.0)      # 제어 루프 페이싱만
    # 전진(크루즈)
    ap.add_argument("--target_speed_kmh", type=float, default=25.0)
    ap.add_argument("--kp_throttle", type=float, default=0.05)
    # KUKSA
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)
    ap.add_argument("--steer_key", default="Vehicle.ADAS.LK.Steering")
    ap.add_argument("--log_every", type=float, default=0.5)
    args = ap.parse_args()

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world = client.get_world()

    ego = find_actor(world, role=args.ego_role, wait_sec=15.0)
    lead = find_actor(world, role=args.lead_role, wait_sec=5.0)
    if ego is None:
        return

    # ego는 TM 개입 금지
    try:
        ego.set_autopilot(False)
    except Exception:
        pass

    # lead = TM Autopilot ON (Env가 tick 전담)
    if lead is not None:
        try:
            tm = client.get_trafficmanager(args.tm_port)
            try:
                tm.set_synchronous_mode(True)  # 월드 tick은 Env 담당
            except Exception:
                pass
            lead.set_autopilot(True, args.tm_port)
            if hasattr(tm, "auto_lane_change"):
                tm.auto_lane_change(lead, False)
            # 살짝 느리게
            try:
                if hasattr(tm, "vehicle_percentage_speed_difference"):
                    tm.vehicle_percentage_speed_difference(lead, 10.0)
                elif hasattr(tm, "set_percentage_speed_difference"):
                    tm.set_percentage_speed_difference(lead, 10.0)
            except Exception:
                pass
            print("[INFO] Lead autopilot ON via TM")
        except Exception as e:
            print("[TM] lead setup failed:", e)
    else:
        print("[WARN] No 'lead' vehicle found.")

    # --- KUKSA ---
    kuksa = KUKSA(args.kuksa_host, args.kuksa_port)

    # --- 상태 변수 ---
    # 전진(스로틀만)
    tgt = float(args.target_speed_kmh)
    kp = float(args.kp_throttle)
    ema_v = tgt
    alpha_v = 0.2
    thr_ema = 0.0
    alpha_thr = 0.25

    last_log = 0.0
    dt = 1.0 / max(1.0, args.hz)
    print(f"[RUN] EGO: throttle-only ≈{tgt:.1f} km/h, STEER=KUKSA RAW '{args.steer_key}' | LEAD: TM ON")
    try:
        while True:
            t0 = time.time()

            # --- 전진: 스로틀만 ---
            v = speed_kmh(ego)
            ema_v = (1.0 - alpha_v) * ema_v + alpha_v * v
            err_v = tgt - ema_v
            raw_thr = clamp(kp * err_v, 0.0, 1.0)
            thr_ema = (1.0 - alpha_thr) * thr_ema + alpha_thr * raw_thr
            throttle = clamp(thr_ema, 0.0, 1.0)
            brake = 0.0

            # --- 조향: KUKSA RAW 그대로 ---
            raw = kuksa.get(args.steer_key)
            steer_cmd = 0.0
            if raw is not None:
                val = _to_float_safe(raw)
                if val is not None:
                    steer_cmd = clamp(val, -1.0, 1.0)
                else:
                    # 타입 미스매치 시 1회 경고 (너무 시끄럽지 않게)
                    # 필요하면 주석 해제해서 디버그 강화
                    # print(f"[KUKSA] WARN: cannot convert to float: type={type(raw)} value={raw!r}")
                    steer_cmd = 0.0

            ego.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=steer_cmd,
                hand_brake=False, reverse=False, manual_gear_shift=False
            ))

            # 디버그 로그
            now = time.time()
            if now - last_log >= max(0.1, args.log_every):
                print(f"[CTL] v={v:5.1f} km/h | thr={throttle:.2f} | steer_raw={steer_cmd:+.3f}")
                last_log = now

            # 페이싱(Env tick에 종속)
            remain = dt - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    except Exception:
        traceback.print_exc()
    finally:
        try:
            ego.set_autopilot(False)
        except Exception:
            pass
        print("[CLEAN] control.py passive exit")


if __name__ == "__main__":
    main()
