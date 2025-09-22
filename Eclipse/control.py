#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control.py

- KUKSA Databroker에서 제어 명령을 읽어 CARLA ego vehicle에 적용하는 템플릿.
- Decision 노드가 KUKSA에 아래 key들로 쓰면 바로 반영됩니다:
    Vehicle.Controls.Throttle   (float 0.0..1.0)
    Vehicle.Controls.Brake      (float 0.0..1.0)
    Vehicle.Controls.Steering   (float -1.0..1.0)   # -: left, +: right
    Vehicle.Controls.Automatic  (bool)             # True -> 자동 제어 허용

- Decision이 없을 경우 fallback: Vehicle.ADAS.ACC.Distance/RelSpeed 를 읽어
  간단한 속도 제어(very basic ACC)를 수행합니다.

- 참고: Env.py 와 decision_LK.py 와 함께 통합하여 사용하세요.
  Env.py는 레이더 → KUKSA publish 구성이 포함되어 있습니다. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
"""

import time
import math
import argparse
import traceback

# CARLA
import carla

# Attempt to import KUKSA Databroker client(s). Env.py 에서 시도한 방식과 호환되도록 여러 옵션 시도
try:
    # some SDKs provide DataBrokerClient
    from kuksa_client.grpc import DataBrokerClient
    _KUKSA_CLIENT_TYPE = "DataBrokerClient"
except Exception:
    try:
        # alternative naming
        from kuksa_client.grpc import VSSClient as DataBrokerClient
        _KUKSA_CLIENT_TYPE = "VSSClient"
    except Exception:
        DataBrokerClient = None
        _KUKSA_CLIENT_TYPE = None

def find_ego_and_wait(world, wait_sec=20.0, poll_hz=10.0):
    """
    role_name=='ego' 인 차량을 최대 wait_sec 동안 기다렸다가 반환.
    동기모드에서 Env.py가 tick을 돌리므로 여기서는 wait_for_tick()으로 프레임을 기다리며
    주기적으로 목록을 재조회한다.
    """
    deadline = time.time() + max(0.0, wait_sec)
    period = 1.0 / max(1.0, poll_hz)

    # 첫 스냅샷을 보장: 동기 월드에서는 프레임이 진행되어야 신규 액터가 보이는 경우가 있음
    try:
        world.wait_for_tick(timeout=2.0)
    except Exception:
        pass

    last_list_print = 0.0
    while time.time() < deadline:
        # 최신 스냅샷 대기(Env.py가 world.tick() 하므로 프레임이 들어오면 깨어남)
        try:
            world.wait_for_tick(timeout=1.0)
        except Exception:
            pass

        actors = world.get_actors().filter("vehicle.*")
        for v in actors:
            if v.attributes.get("role_name", "") == "ego":
                print(f"[INFO] Found ego (id={v.id})")
                return v

        now = time.time()
        if now - last_list_print > 3.0:
            print("[DBG] ego not visible yet. vehicles:")
            for v in actors:
                print(f"  id={v.id:>5} type={v.type_id:<28} role={v.attributes.get('role_name')}")
            last_list_print = now

        time.sleep(period)

    print("[ERR] Timed out waiting for role_name='ego'.")
    return None

# ------------------------
# 유틸: 안전 범위 적용
# ------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------------
# KUKSA 래퍼 클래스 (유연성)
# ------------------------
class KuksaWrapper:
    def __init__(self, host="127.0.0.1", port=55555):
        self.client = None
        self.host = host
        self.port = port
        if DataBrokerClient is None:
            print("[KUKSA] kuksa_client not available; KUKSA read/write will be disabled.")
            return
        try:
            # Env.py와 유사한 시도
            # Some SDK constructors accept (host, port), others accept endpoint string; try multiple
            try:
                self.client = DataBrokerClient(host, port)
            except TypeError:
                # maybe constructor signature different
                self.client = DataBrokerClient(f"{host}:{port}")
            try:
                # many SDKs require explicit connect()
                self.client.connect()
            except Exception:
                pass
            print(f"[KUKSA] Connected using {DataBrokerClient.__name__}")
        except Exception as e:
            print("[KUKSA] Failed to init client:", e)
            self.client = None

    def get(self, key):
        """KUKSA에서 key의 값을 읽어 반환. 실패시 None."""
        if self.client is None:
            return None
        try:
            # 여러 SDK 버전에 대비: get / get_value / get_datapoint 등 시도
            if hasattr(self.client, "get"):
                return self.client.get(key)
            if hasattr(self.client, "get_value"):
                return self.client.get_value(key)
            if hasattr(self.client, "read"):
                return self.client.read(key)
            # fallback: 일부에서는 attribute style
            return None
        except Exception as e:
            # 디버깅 위해 로그 남김
            # print("[KUKSA] get error:", e)
            return None

    def set(self, key, value):
        if self.client is None:
            return False
        try:
            if hasattr(self.client, "set"):
                self.client.set(key, value)
                return True
            if hasattr(self.client, "put"):
                self.client.put(key, value)
                return True
            if hasattr(self.client, "write"):
                self.client.write(key, value)
                return True
        except Exception as e:
            print("[KUKSA] set failed:", e)
        return False

# ------------------------
# 기본 ACC (fallback) 컨트롤러
# very simple proportional controller that turns distance -> throttle
# ------------------------
class SimpleACC:
    def __init__(self, desired_distance=20.0, max_throttle=0.6):
        self.desired_distance = desired_distance
        self.max_throttle = max_throttle
        self.kp = 0.08  # 거리에 대한 비례 이득 (튜닝 필요)
        self.last_throttle = 0.0
        self.last_brake = 0.0

    def compute(self, distance, rel_speed, ego_speed):
        """
        Inputs:
          distance: m (float or None)
          rel_speed: m/s (positive => lead receding, negative => approaching)
          ego_speed: m/s
        Returns:
          throttle (0..1), brake (0..1), steering (0..1 placeholder)
        """
        if distance is None or math.isinf(distance):
            # 타겟 없음 -> 크루즈(약한 토틀 유지)
            return 0.25, 0.0, 0.0

        # 거리오차: 멀면 가속, 가깝으면 감속
        err = distance - self.desired_distance
        thr = clamp(self.kp * err, 0.0, self.max_throttle)

        # 상대 속도 보정: 너무 빨리 접근하면 브레이크
        brk = 0.0
        if rel_speed < -1.5:  # approaching fast -> brake
            brk = clamp(-rel_speed * 0.2, 0.0, 1.0)
            thr = 0.0

        # smoothing
        thr = 0.6 * self.last_throttle + 0.4 * thr
        brk = 0.6 * self.last_brake + 0.4 * brk
        self.last_throttle = thr
        self.last_brake = brk
        return thr, brk, 0.0

# ------------------------
# main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="CARLA host")
    ap.add_argument("--port", type=int, default=2000, help="CARLA port")
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)
    ap.add_argument("--ego_role", default="ego", help="role_name of ego vehicle to control")
    ap.add_argument("--hz", type=float, default=20.0, help="control loop Hz")
    ap.add_argument("--timeout_no_cmd", type=float, default=1.0, help="seconds to wait before zeroing commands")
    ap.add_argument("--desired_distance", type=float, default=20.0, help="ACC desired following dist (m) fallback")
    args = ap.parse_args()

    # CARLA connect
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    # find ego actor by role_name
    ego = find_ego_and_wait(world, wait_sec=20.0, poll_hz=10.0)
    if ego is None:
        return

    print(f"[INFO] controlling actor id={ego.id} role={args.ego_role}")

    # KUKSA wrapper
    kuksa = KuksaWrapper(host=args.kuksa_host, port=args.kuksa_port)

    acc = SimpleACC(desired_distance=args.desired_distance)

    loop_dt = 1.0 / max(1.0, args.hz)
    last_cmd_ts = 0.0

    try:
        while True:
            t0 = time.time()

            # ---------------------------------------
            # 1) Try reading explicit control commands from KUKSA
            #    -> Decision 노드가 아래 경로에 값을 쓰면 바로 사용
            # ---------------------------------------
            thr = None
            brk = None
            str_cmd = None
            auto_on = True

            if kuksa.client is not None:
                try:
                    # SDK마다 반환 형식이 다르므로 안전하게 unwrap
                    v_thr = kuksa.get("Vehicle.Controls.Throttle")
                    v_brk = kuksa.get("Vehicle.Controls.Brake")
                    v_str = kuksa.get("Vehicle.Controls.Steering")
                    v_auto = kuksa.get("Vehicle.Controls.Automatic")
                    # unwrap value if wrapped
                    def _unwrap(x):
                        if x is None:
                            return None
                        if isinstance(x, dict) and "value" in x:
                            return x["value"]
                        return x
                    thr = _unwrap(v_thr)
                    brk = _unwrap(v_brk)
                    str_cmd = _unwrap(v_str)
                    v_auto = _unwrap(v_auto)
                    if v_auto is not None:
                        auto_on = bool(v_auto)
                except Exception:
                    thr = brk = str_cmd = None

            have_explicit = (thr is not None and brk is not None and str_cmd is not None and auto_on)

            if have_explicit:
                # 안전 범위 적용
                thr = clamp(float(thr), 0.0, 1.0)
                brk = clamp(float(brk), 0.0, 1.0)
                str_cmd = clamp(float(str_cmd), -1.0, 1.0)
                last_cmd_ts = time.time()
            else:
                # ---------------------------------------
                # 2) Fallback: read ACC sensors & compute basic controls
                #    (Env.py에서 레이더가 KUKSA에 publish 하는 키 사용)
                #    Vehicle.ADAS.ACC.Distance, Vehicle.ADAS.ACC.RelSpeed, Vehicle.ADAS.ACC.HasTarget
                # ---------------------------------------
                dist = None
                rel_speed = None
                if kuksa.client is not None:
                    try:
                        _d = kuksa.get("Vehicle.ADAS.ACC.Distance")
                        _r = kuksa.get("Vehicle.ADAS.ACC.RelSpeed")
                        _h = kuksa.get("Vehicle.ADAS.ACC.HasTarget")
                        def _u(x):
                            if x is None:
                                return None
                            if isinstance(x, dict) and "value" in x:
                                return x["value"]
                            return x
                        dist = _u(_d)
                        rel_speed = _u(_r)
                        has_target = _u(_h)
                        if has_target in (False, 0, "False", "false", None):
                            dist = None
                    except Exception:
                        dist = rel_speed = None

                # get ego speed from CARLA
                v = ego.get_velocity()
                ego_speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

                thr, brk, str_cmd = acc.compute(dist, rel_speed, ego_speed)
                # steering fallback: keep current steering 0
                str_cmd = 0.0
                last_cmd_ts = time.time()

            # ---------------------------------------
            # 3) Safety timeout: 제어 명령이 오래 없으면 정지
            # ---------------------------------------
            if (time.time() - last_cmd_ts) > args.timeout_no_cmd:
                # 안전정지(또는 낮은 토틀)
                thr = 0.0
                brk = 0.8  # 강한 제동
                str_cmd = 0.0

            # ---------------------------------------
            # 4) Apply control to CARLA ego
            #    - CARLA control expects throttle 0..1, brake 0..1, steer -1..1
            # ---------------------------------------
            try:
                # build VehicleControl
                ctrl = carla.VehicleControl()
                ctrl.throttle = float(clamp(thr, 0.0, 1.0))
                ctrl.brake = float(clamp(brk, 0.0, 1.0))
                ctrl.steer = float(clamp(str_cmd, -1.0, 1.0))
                # optional: autopilot flag off when using direct control
                ctrl.hand_brake = False
                ctrl.reverse = False
                ego.apply_control(ctrl)
            except Exception as e:
                print("[ERR] apply_control:", e)

            # ---------------------------------------
            # 5) (선택) 현재 적용한 제어값을 KUKSA에 다시 기록 (피드백)
            # ---------------------------------------
            try:
                if kuksa.client is not None:
                    kuksa.set("Vehicle.Controls.AppliedThrottle", ctrl.throttle)
                    kuksa.set("Vehicle.Controls.AppliedBrake", ctrl.brake)
                    kuksa.set("Vehicle.Controls.AppliedSteering", ctrl.steer)
            except Exception:
                pass

            # loop pacing
            t1 = time.time()
            dt = t1 - t0
            sleep_for = loop_dt - dt
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received, cleaning up...")
    except Exception:
        traceback.print_exc()
    finally:
        print("[CLEAN] control.py exiting.")

if __name__ == "__main__":
    main()
