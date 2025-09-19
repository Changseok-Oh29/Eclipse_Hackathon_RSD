#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kuksa_slip_estimator.py
# CARLA 차량에 센서만 부착 → 슬립 기반 노면 추정 → Kuksa(VSS) 퍼블리시
# (옵션) zenoh JSON 병행 퍼블리시 가능

import argparse, math, time, json, socket, collections, traceback
import carla

# --- Kuksa ---
from kuksa_client.grpc import VSSClient, Datapoint

# --- (옵션) zenoh 병행 발행 ---
try:
    import zenoh
except Exception:
    zenoh = None


def rad(d): return d * math.pi / 180.0
def clamp(v, a, b): return max(a, min(b, v))


def find_vehicle(world, attach_id=None, prefer_roles=("hero", "ego")):
    actors = world.get_actors().filter("vehicle.*")

    if attach_id is not None:
        a = world.get_actor(attach_id)
        if a and "vehicle" in a.type_id:
            return a
        raise RuntimeError(f"[ERR] actor_id {attach_id} is not a vehicle or not found.")

    for role in prefer_roles:
        for a in actors:
            if a.attributes.get("role_name", "") == role:
                return a

    best, best_v2 = None, -1.0
    for a in actors:
        v = a.get_velocity()
        v2 = v.x * v.x + v.y * v.y + v.z * v.z
        if v2 > best_v2:
            best, best_v2 = a, v2
    if best:
        return best
    raise RuntimeError("[ERR] no vehicle found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-host", default="127.0.0.1")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--town", default=None)
    ap.add_argument("--hz", type=float, default=20.0)

    ap.add_argument("--role", default="ego")
    ap.add_argument("--attach", type=int, default=None, help="붙을 vehicle actor_id")

    # Kuksa
    ap.add_argument("--kuksa-host", default="127.0.0.1")
    ap.add_argument("--kuksa-port", type=int, default=55555)

    # (옵션) 제노/UDP 병행
    ap.add_argument("--zkey", default="", help="zenoh publish key (빈 문자열이면 비활성)")
    ap.add_argument("--udp", default="", help="host:port (옵션)")

    # 튠 파라미터
    ap.add_argument("--thr2acc", type=float, default=1.2, help="가속 기대치 계수 k_thr")
    ap.add_argument("--brk2dec", type=float, default=4.5, help="감속 기대치 계수 k_brk")
    ap.add_argument("--min_v_for_quality", type=float, default=5.0, help="품질 가중 활성 최소 속도[m/s]")
    ap.add_argument("--low_v", type=float, default=2.0, help="저속 게이팅 임계[m/s]")
    ap.add_argument("--evt_thr", type=float, default=0.2, help="토크 이벤트(스로틀/브레이크) 임계")
    ap.add_argument("--place_friction", action="store_true", help="전방 저마찰 패치 생성")
    args = ap.parse_args()

    # --- CARLA 연결 ---
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world = client.get_world()
    if args.town and world.get_map().name.split("/")[-1] != args.town:
        world = client.load_world(args.town)
        time.sleep(0.3)

    bp = world.get_blueprint_library()

    # === 기존 차량 찾기 (스폰 안 함) ===
    ego = find_vehicle(world, attach_id=args.attach, prefer_roles=("hero", "ego", args.role))
    print(f"[INFO] attached to vehicle id={ego.id}, role={ego.attributes.get('role_name','')}")

    # === 센서들 (측정 전용) ===
    imu_bp = bp.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", f"{1.0 / args.hz:.4f}")
    imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=1.0)), attach_to=ego)

    ws_sensor = None
    wodo_sensor = None
    try:
        ws_bp = bp.find("sensor.other.wheel_slip")
        ws_bp.set_attribute("sensor_tick", f"{1.0 / args.hz:.4f}")
        ws_sensor = world.spawn_actor(ws_bp, carla.Transform(), attach_to=ego)
    except Exception:
        pass
    try:
        wodo_bp = bp.find("sensor.other.wheel_odometry")
        wodo_bp.set_attribute("sensor_tick", f"{1.0 / args.hz:.4f}")
        wodo_sensor = world.spawn_actor(wodo_bp, carla.Transform(), attach_to=ego)
    except Exception:
        pass

    friction_actor = None
    if args.place_friction:
        try:
            ft_bp = bp.find("static.trigger.friction")
            ft_bp.set_attribute("friction", "0.25")
            ft_bp.set_attribute("scale", "35")
            loc = ego.get_location()
            fwd = ego.get_transform().get_forward_vector()
            place = carla.Location(loc.x + fwd.x * 40.0, loc.y + fwd.y * 40.0, loc.z)
            friction_actor = world.spawn_actor(ft_bp, carla.Transform(place))
            print("[INFO] low-µ patch placed at +40m")
        except Exception:
            print("[WARN] friction trigger not available.")

    # === 버퍼 & 통신 (옵션용) ===
    N = max(1, int(args.hz * 0.5))   # 약 0.5초 이동평균
    ay_buf, ax_buf = collections.deque(maxlen=N), collections.deque(maxlen=N)
    ws_buf, vr_buf = collections.deque(maxlen=N), collections.deque(maxlen=N)
    label = "unknown"

    udp_sock, udp_addr = None, None
    if args.udp:
        host, port = args.udp.split(":")
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr = (host, int(port))

    z_session, z_pub = None, None
    if zenoh is not None and args.zkey:
        try:
            z_session = zenoh.open(zenoh.Config())
            z_pub = z_session.declare_publisher(args.zkey)
            print(f"[ZENOH] publish → '{args.zkey}' (optional)")
        except Exception as e:
            print("[ZENOH] open failed:", e)

    # === Kuksa 연결 ===
    kc = VSSClient(args.kuksa_host, args.kuksa_port)
    kc.connect()
    print(f"[KUKSA] connected @ {args.kuksa_host}:{args.kuksa_port}")

    # --- 콜백 ---
    def on_imu(ev):
        ax_buf.append(float(ev.accelerometer.x))
        ay_buf.append(abs(float(ev.accelerometer.y)))

    def on_ws(ev):
        try:
            ws_buf.append(abs(float(ev.slip)))
        except Exception:
            pass

    def on_wodo(ev):
        try:
            sp = getattr(ev, "speed", None)
            if sp is not None:
                vr_buf.append(float(sp))
        except Exception:
            pass

    imu.listen(on_imu)
    if ws_sensor:  ws_sensor.listen(on_ws)
    if wodo_sensor: wodo_sensor.listen(on_wodo)

    dt = 1.0 / args.hz
    eps = 1e-3
    k_thr, k_brk = args.thr2acc, args.brk2dec
    last_print = 0.0

    try:
        while True:
            world.tick()
            sim_ts = world.get_snapshot().timestamp.elapsed_seconds
            tr = ego.get_transform()
            vel = ego.get_velocity()
            yaw = rad(tr.rotation.yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            vx = c * vel.x + s * vel.y
            vy = -s * vel.x + c * vel.y
            v = math.hypot(vx, vy)

            alpha = math.degrees(math.atan2(vy, abs(vx) + eps))
            ax = sum(ax_buf) / len(ax_buf) if ax_buf else 0.0
            ay = sum(ay_buf) / len(ay_buf) if ay_buf else 0.0

            ctrl = ego.get_control()
            thr, brk, steer = float(ctrl.throttle), float(ctrl.brake), float(abs(ctrl.steer))

            a_exp = k_thr * thr - k_brk * brk
            long_residual = a_exp - ax

            S_ws = sum(ws_buf) / len(ws_buf) if ws_buf else 0.0
            v_wodo = sum(vr_buf) / len(vr_buf) if vr_buf else None

            kappa = None
            if v_wodo is not None and (v_wodo > 0.5 or v > 0.5):
                denom = max(max(v_wodo, v), eps)
                kappa = (v_wodo - v) / denom  # [-1, 1] 근사

            score_lat = abs(alpha) / 6.0
            if thr > 0.5:
                lack = (k_thr * thr - ax)
                score_long = clamp((lack - 0.8) / 2.0, 0.0, 1.2)
                if v < 5.0:
                    score_long = clamp((lack - 0.5) / 1.5, 0.0, 1.2)
            elif thr > 0.2:
                score_long = clamp((long_residual - 1.2) / 2.5, 0.0, 1.2)
            elif brk > 0.2:
                score_long = clamp(((-long_residual) - 2.0) / 4.0, 0.0, 1.2)
            else:
                score_long = 0.0

            score_ws = clamp((S_ws - 0.06) / 0.12, 0.0, 1.2) if S_ws > 0 else 0.0
            if kappa is not None and (thr > args.evt_thr or brk > args.evt_thr):
                score_ws = max(score_ws, clamp((abs(kappa) - 0.06) / 0.12, 0.0, 1.2))

            score = 0.5 * score_lat + 0.4 * score_long + 0.3 * score_ws

            # 상태/신뢰도
            if v < args.low_v and thr < 0.1 and brk < 0.05:
                state, conf = "unknown", 0.3
            else:
                if score > 0.9:
                    state, conf = "ice", min(0.95, 0.6 + 0.4 * score)
                elif score > 0.45:
                    state, conf = "wet", min(0.9, 0.5 + 0.3 * score)
                else:
                    state, conf = "dry", (0.6 if score < 0.25 else 0.55)

            # 품질 (융합 가중)
            quality = 0.0
            if v > args.min_v_for_quality and (thr > args.evt_thr or brk > args.evt_thr):
                q0 = 0.6
                q_evt = min(0.4, 0.4 * max(thr, brk))
                quality = min(1.0, q0 + q_evt)

            # Kuksa 업데이트(배치)
            updates = {
                "Vehicle.Private.Slip.State":      Datapoint(state),
                "Vehicle.Private.Slip.Quality":    Datapoint(float(quality)),
                "Vehicle.Private.Slip.Confidence": Datapoint(float(conf)),
                "Vehicle.Private.Slip.Ts":         Datapoint(float(sim_ts)),
                "Vehicle.Private.Slip.Metrics.v":                 Datapoint(float(v)),
                "Vehicle.Private.Slip.Metrics.vx":                Datapoint(float(vx)),
                "Vehicle.Private.Slip.Metrics.vy":                Datapoint(float(vy)),
                "Vehicle.Private.Slip.Metrics.alpha_deg":         Datapoint(float(alpha)),
                "Vehicle.Private.Slip.Metrics.ax_mean":           Datapoint(float(ax)),
                "Vehicle.Private.Slip.Metrics.ay_abs_mean":       Datapoint(float(ay)),
                "Vehicle.Private.Slip.Metrics.long_residual":     Datapoint(float(long_residual)),
                "Vehicle.Private.Slip.Metrics.wheel_slip_mean":   Datapoint(float(S_ws)),
                "Vehicle.Private.Slip.Metrics.wheel_odo_v_mean":  Datapoint(float(v_wodo) if v_wodo is not None else 0.0),
                "Vehicle.Private.Slip.Metrics.kappa_est":         Datapoint(float(kappa) if kappa is not None else 0.0),
            }
            try:
                # Databroker는 set_current_values(v2) 사용 (브로커 실행 시 --enable-databroker-v1 필요)
                kc.set_current_values(updates)
            except Exception:
                traceback.print_exc()

            # (옵션) 콘솔/UDP/zenoh 병행 발행
            msg = {
                "src": "slip_estimator",
                "ts": sim_ts,
                "state": state,
                "confidence": round(conf, 3),
                "quality": round(quality, 3),
                "metrics": {
                    "v": round(v, 2),
                    "vx": round(vx, 2),
                    "vy": round(vy, 2),
                    "alpha_deg": round(alpha, 2),
                    "ax_mean": round(ax, 3),
                    "ay_abs_mean": round(ay, 3),
                    "long_residual": round(long_residual, 3),
                    "wheel_slip_mean": round(S_ws, 4),
                    "wheel_odo_v_mean": (round(v_wodo, 2) if v_wodo is not None else None),
                    "kappa_est": (round(kappa, 4) if kappa is not None else None)
                },
                "control": {"thr": round(thr, 2), "brk": round(brk, 2), "steer": round(steer, 2)}
            }
            line = json.dumps(msg, ensure_ascii=False)
            print(line)

            if args.udp:
                try: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(line.encode("utf-8"), udp_addr)
                except Exception: pass
            if z_pub is not None:
                try: z_pub.put(line.encode("utf-8"))
                except Exception: pass

            # 진행 로그 주기 조절
            t = time.time()
            if t - last_print > 1.0:
                last_print = t

            time.sleep(dt)

    finally:
        if ws_sensor:
            try: ws_sensor.stop(); ws_sensor.destroy()
            except Exception: pass
        if wodo_sensor:
            try: wodo_sensor.stop(); wodo_sensor.destroy()
            except Exception: pass
        try: imu.stop(); imu.destroy()
        except Exception: pass
        if friction_actor:
            try: friction_actor.destroy()
            except Exception: pass
        if z_session:
            try: z_session.close()
            except Exception: pass
        try: kc.disconnect()
        except Exception: pass


if __name__ == "__main__":
    main()
