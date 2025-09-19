#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sl2.py — CARLA Slip Estimator (env.py 호환 패시브 모드, 심시간 통일)

핵심:
- env.py가 world.tick()을 주도 → 본 스크립트는 wait_for_tick()으로 동작 (기본)
- 차량 탐색: attach_id → role_name('ego' 우선) → 가장 빠른 차량, N초 대기 후 실패
- 옵션: 차량이 없으면 직접 스폰(--spawn-if-missing)
- 센서 sensor_tick을 world.fixed_delta_seconds에 정렬
- Slip.Ts는 심시간(sim_ts)만 사용 + (sync & fixed_dt)일 때 Δt 격자에 스냅(quantize)
- 벽시계 시간 사용 금지 (융합 실패 방지)
- Kuksa VSS로 Slip 상태/품질/신뢰도 및 메트릭 업데이트
- (옵션) UDP/Zenoh 보조 퍼블리시

실행 예:
1) env.py 먼저 실행 (동기식 월드 주도)
   python env.py --fps 20 --display 1
2) 슬립 추정기 실행
   python sl2.py --carla-host 127.0.0.1 --carla-port 2000 \
                 --kuksa-host 127.0.0.1 --kuksa-port 55555
   # 차량 없으면 직접 스폰
   python sl2.py --spawn-if-missing
"""

import argparse, math, time, json, socket, collections, traceback
import carla

# Kuksa gRPC 클라이언트
try:
    from kuksa_client.grpc import VSSClient, Datapoint
except Exception as e:
    raise RuntimeError("kuksa_client.grpc 가 필요합니다. `pip install kuksa-client`") from e

# Zenoh(선택)
try:
    import zenoh
except Exception:
    zenoh = None


# ---------- 유틸 ----------
def rad(d): return d * math.pi / 180.0
def clamp(v, a, b): return max(a, min(b, v))

def find_vehicle(world, attach_id=None, prefer_roles=("hero", "ego"), wait_sec=10.0):
    """
    월드에서 차량 탐색: attach_id → prefer_roles → 가장 빠른 차량
    없으면 wait_sec 동안 재시도 후 예외
    """
    t0 = time.time()
    last_err = None
    while True:
        try:
            actors = world.get_actors().filter("vehicle.*")

            # 1) attach_id 최우선
            if attach_id is not None:
                try:
                    a = world.get_actor(attach_id)
                    if a and "vehicle" in a.type_id:
                        return a
                    raise RuntimeError(f"actor_id {attach_id} not a vehicle or missing")
                except Exception as e:
                    last_err = e  # 아래 탐색 계속

            # 2) role_name 우선
            for role in prefer_roles:
                for a in actors:
                    if a.attributes.get("role_name", "") == role:
                        return a

            # 3) 가장 빠른 차량
            best, best_v2 = None, -1.0
            for a in actors:
                v = a.get_velocity()
                v2 = v.x*v.x + v.y*v.y + v.z*v.z
                if v2 > best_v2:
                    best, best_v2 = a, v2
            if best:
                return best

            # 재시도
            if time.time() - t0 >= wait_sec:
                break
            try:
                world.wait_for_tick()  # sync일 때 자연스럽게 대기
            except Exception:
                time.sleep(0.05)
        except Exception as e:
            last_err = e
            if time.time() - t0 >= wait_sec:
                break
            time.sleep(0.05)

    if last_err:
        raise RuntimeError(f"[ERR] no vehicle found within {wait_sec:.1f}s (last={last_err})")
    else:
        raise RuntimeError(f"[ERR] no vehicle found within {wait_sec:.1f}s")

def try_spawn_ego(world, role_name="ego"):
    """
    월드에 차량이 없을 때 간단히 하나 스폰(빈 스폰포인트가 있어야 함)
    """
    bp = world.get_blueprint_library()
    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', role_name)
    sps = world.get_map().get_spawn_points()
    for tf in sps:
        a = world.try_spawn_actor(ego_bp, tf)
        if a is not None:
            return a
    return None


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    # CARLA
    ap.add_argument("--carla-host", default="127.0.0.1")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--attach", type=int, default=None, help="붙을 vehicle actor_id")
    ap.add_argument("--role", default="ego", help="우선 탐색할 role_name")
    ap.add_argument("--wait-sec", type=float, default=10.0, help="차량 탐색 대기 시간(초)")
    ap.add_argument("--spawn-if-missing", action="store_true",
                    help="월드에 차량이 없으면 ego를 직접 스폰")
    ap.add_argument("--hz", type=float, default=20.0,
                    help="비동기 월드일 때 내부 주파수(동기식이면 무시)")
    ap.add_argument("--drive-tick", action="store_true",
                    help="(비권장) 본 프로세스가 world.tick() 호출 — 동기식 단독 실행시에만 사용")

    # Kuksa
    ap.add_argument("--kuksa-host", default="127.0.0.1")
    ap.add_argument("--kuksa-port", type=int, default=55555)

    # 보조 퍼블리시(옵션)
    ap.add_argument("--udp", default="", help="host:port 로 JSON 상태 전송(옵션)")
    ap.add_argument("--zkey", default="", help="zenoh key (예: 'carla/slip/debug')")

    # 튠 파라미터
    ap.add_argument("--thr2acc", type=float, default=1.2)
    ap.add_argument("--brk2dec", type=float, default=4.5)
    ap.add_argument("--min_v_for_quality", type=float, default=5.0)
    ap.add_argument("--low_v", type=float, default=2.0)
    ap.add_argument("--evt_thr", type=float, default=0.2)

    # (선택) 소폭 바이어스
    ap.add_argument("--ts-bias", type=float, default=0.0,
                    help="Slip.Ts 에 더할 상수 바이어스(초). 기본 0.0")
    args = ap.parse_args()

    # --- CARLA 연결 (env.py 설정은 변경하지 않음) ---
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world = client.get_world()
    settings = world.get_settings()
    sync = settings.synchronous_mode
    fixed_dt = settings.fixed_delta_seconds if settings.fixed_delta_seconds else None
    print(f"[CARLA] sync={sync} fixed_dt={fixed_dt}")

    # --- 차량 attach ---
    try:
        ego = find_vehicle(world,
                           attach_id=args.attach,
                           prefer_roles=("hero", "ego", args.role),
                           wait_sec=args.wait_sec)
    except RuntimeError as e:
        if args.spawn_if_missing:
            print("[INFO] no vehicle found; trying to spawn one...")
            ego = try_spawn_ego(world, role_name=args.role)
            if ego is None:
                raise RuntimeError("[ERR] failed to spawn ego as fallback") from e
            print(f"[INFO] spawned ego id={ego.id}")
        else:
            raise
    print(f"[INFO] attached to vehicle id={ego.id}, role={ego.attributes.get('role_name','')}")

    # --- 센서 부착 (측정/계산 전용) ---
    hz = args.hz
    sensor_dt = fixed_dt if (sync and fixed_dt) else (1.0 / max(1.0, hz))

    bp = world.get_blueprint_library()

    imu_bp = bp.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", f"{sensor_dt:.6f}")
    imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=1.0)), attach_to=ego)

    ws_sensor = None
    wodo_sensor = None
    try:
        ws_bp = bp.find("sensor.other.wheel_slip")
        ws_bp.set_attribute("sensor_tick", f"{sensor_dt:.6f}")
        ws_sensor = world.spawn_actor(ws_bp, carla.Transform(), attach_to=ego)
    except Exception:
        print("[WARN] wheel_slip sensor not available.")

    try:
        wodo_bp = bp.find("sensor.other.wheel_odometry")
        wodo_bp.set_attribute("sensor_tick", f"{sensor_dt:.6f}")
        wodo_sensor = world.spawn_actor(wodo_bp, carla.Transform(), attach_to=ego)
    except Exception:
        print("[WARN] wheel_odometry sensor not available.")

    # --- 버퍼 & 통신 ---
    N = max(1, int((1.0 / sensor_dt) * 0.5)) if sensor_dt else max(1, int(hz * 0.5))
    ay_buf, ax_buf = collections.deque(maxlen=N), collections.deque(maxlen=N)
    ws_buf, vr_buf = collections.deque(maxlen=N), collections.deque(maxlen=N)

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

    # --- Kuksa 연결 ---
    kc = VSSClient(args.kuksa_host, args.kuksa_port)
    kc.connect()
    print(f"[KUKSA] connected @ {args.kuksa_host}:{args.kuksa_port}")

    # --- 센서 콜백 ---
    def on_imu(ev):
        # Carla IMU 단위: m/s^2
        ax_buf.append(float(ev.accelerometer.x))
        ay_buf.append(abs(float(ev.accelerometer.y)))

    def on_ws(ev):
        # Carla wheel_slip: 평균 slip 정량치(버전에 따라 필드 상이 가능 → 예외면 무시)
        try: ws_buf.append(abs(float(ev.slip)))
        except Exception: pass

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

    # --- 루프 ---
    eps = 1e-3
    k_thr, k_brk = args.thr2acc, args.brk2dec
    dt_fallback = (fixed_dt if (sync and fixed_dt) else (1.0 / max(1.0, hz)))

    try:
        while True:
            # Tick 제어
            if sync:
                if args.drive_tick:
                    world.tick()
                    snapshot = world.get_snapshot()
                else:
                    snapshot = world.wait_for_tick()
            else:
                time.sleep(dt_fallback)
                snapshot = world.get_snapshot()

            # 1) sim_ts 생성 (심시간만 사용)
            sim_ts = float(snapshot.timestamp.elapsed_seconds)

            # 2) (동기식 & 고정 Δt)일 때 격자에 스냅(quantize) → 카메라 프레임과 정확히 일치
            if sync and fixed_dt:
                dt = float(fixed_dt)
                n = int(round(sim_ts / dt))
                sim_ts = n * dt
                sim_ts = round(sim_ts, 6)  # 부동소수 미세오차 정리

            # 3) (선택) 소폭 바이어스 적용
            if args.ts_bias != 0.0:
                sim_ts = round(sim_ts + args.ts_bias, 6)

            # 차량 상태 추출
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

            # 점수(경험적 조합)
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

            # 품질(이벤트 기반 가중)
            quality = 0.0
            if v > args.min_v_for_quality and (thr > args.evt_thr or brk > args.evt_thr):
                q0 = 0.6
                q_evt = min(0.4, 0.4 * max(thr, brk))
                quality = min(1.0, q0 + q_evt)

            # --- Kuksa 업데이트 (심시간만) ---
            updates = {
                "Vehicle.Private.Slip.State":      Datapoint(state),
                "Vehicle.Private.Slip.Quality":    Datapoint(float(quality)),
                "Vehicle.Private.Slip.Confidence": Datapoint(float(conf)),
                "Vehicle.Private.Slip.Ts":         Datapoint(float(sim_ts)),  # ★ 심시간/격자 스냅 적용
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
                kc.set_current_values(updates)
            except Exception:
                traceback.print_exc()

            # (옵션) 로그/UDP/Zenoh — 심시간만 사용
            msg = {
                "src": "slip_estimator",
                "ts": sim_ts,  # 심시간
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

            if udp_sock and udp_addr:
                try: udp_sock.sendto(line.encode("utf-8"), udp_addr)
                except Exception: pass
            if z_pub is not None:
                try: z_pub.put(line.encode("utf-8"))
                except Exception: pass

    finally:
        # 정리
        try:
            if ws_sensor: ws_sensor.stop(); ws_sensor.destroy()
        except Exception: pass
        try:
            if wodo_sensor: wodo_sensor.stop(); wodo_sensor.destroy()
        except Exception: pass
        try:
            imu.stop(); imu.destroy()
        except Exception: pass
        if z_session:
            try: z_session.close()
            except Exception: pass
        try:
            kc.disconnect()
        except Exception: pass


if __name__ == "__main__":
    main()
