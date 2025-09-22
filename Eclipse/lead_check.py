#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lead_check.py — Pure Pursuit + 시나리오 속도제어 (리드 차량 μ=dry 고정)

변경 요약:
- 시나리오별 단일 날씨 적용(주행 중 전환 제거)
- 시나리오 0: 기존 루트(인덱스 20), 기본 날씨, 목표속도 60 km/h (커브 자동 감속 유지)
- 시나리오 1~3: 두 번째 직선만 사용. 스폰 = 스폰포인트 53에서 차선방향 기준 200 m 뒤 지점
  * S1: 기본 날씨, 60 km/h
  * S2: Heavy Rain, 60 km/h
  * S3: WetNoon, 55 km/h
- 직선 시나리오(1~3) 포함 공통: 650~700 m 선형 감속, 700 m 정지
- 충돌 발생 시 리드 차량 즉시 정지
- 리드 차량 타이어 마찰계수 μ는 항상 dry 유지
- TM/Autopilot/agents 미사용. 기본 spawn_idx=20(시나리오 0에서만 사용)
"""

import sys, time, math, threading, queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import carla

# =========================
# 유틸
# =========================
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def deg_wrap180(a): return (a + 180.0) % 360.0 - 180.0
def norm2d(x, y): return math.sqrt(x*x + y*y)
def dot2d(ax, ay, bx, by): return ax*bx + ay*by

# =========================
# 두 번째 직선 기준 (기본값: 과거 고정 프레임)
# 직선 시나리오에서는 동적으로 재설정
# =========================
DEFAULT_LINE_START = carla.Location(x=343.40,  y=21.69,  z=1.48)  # m 단위로 표기
DEFAULT_LINE_END   = carla.Location(x=-362.11, y=15.80,  z=1.65)

RAIN_SWITCH_S  = 300.0   # (참고) 사용 안 함. 주행 중 날씨 전환 제거
DECEL_START_S  = 650.0   # +650 m : 감속 시작
STOP_S         = 700.0   # +700 m : 정지
MAX_CRUISE_KMH = 60.0    # 선형 감속 상한 기준 속도

class TrackFrame:
    """s 축(직선) 정의. start~end를 따라 s를 계산."""
    def __init__(self, start: carla.Location, end: carla.Location):
        self.set_line(start, end)
    def set_line(self, start: carla.Location, end: carla.Location):
        self.sx, self.sy = start.x, start.y
        self.ex, self.ey = end.x, end.y
        dx, dy = (self.ex - self.sx), (self.ey - self.sy)
        self.L = norm2d(dx, dy)
        if self.L < 1e-6:
            self.ux, self.uy = 1.0, 0.0
        else:
            self.ux, self.uy = dx/self.L, dy/self.L
    def s_on_line(self, p: carla.Location) -> float:
        if self.L < 1e-6: return -1e9
        px, py = p.x - self.sx, p.y - self.sy
        return dot2d(px, py, self.ux, self.uy)

track_frame = TrackFrame(DEFAULT_LINE_START, DEFAULT_LINE_END)

# =========================
# PID (속도 제어용)
# =========================
class PID:
    def __init__(self, kp, ki, kd, i_limit=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0; self.i_limit = i_limit
        self.prev_e = 0.0; self.first = True
    def reset(self):
        self.i = 0.0; self.prev_e = 0.0; self.first = True
    def step(self, e, dt):
        self.i += e*dt; self.i = clamp(self.i, -self.i_limit, self.i_limit)
        d = 0.0 if self.first else (e - self.prev_e)/max(dt, 1e-3)
        self.prev_e = e; self.first = False
        return self.kp*e + self.ki*self.i + self.kd*d

# =========================
# 시나리오
# =========================
@dataclass
class Scenario:
    sid: int
    name: str
    # pattern: (duration_sec, target_kmh)
    pattern: List[Tuple[float, float]]
    loop: bool
    route_type: str  # "mixed"(기존 루트) or "straight"(두 번째 직선)
    weather: Optional[carla.WeatherParameters]  # None=현재 월드 날씨 유지
    target_kmh: float  # 편의상 패턴 대신 단일 목표 속도 사용 시 참고

def get_scenario(sid: int, original_weather: carla.WeatherParameters) -> Scenario:
    # 시나리오 0: 기존 루트, 기본 날씨, 60 km/h
    if sid == 0:
        return Scenario(
            sid=0,
            name="S0",
            pattern=[(9999.0, 60.0)],  # 단일 속도(커브 감속은 별도 로직)
            loop=True,
            route_type="mixed",
            weather=None,  # 기본 날씨 유지
            target_kmh=60.0
        )
    # 시나리오 1: 직선, 기본 날씨, 60 km/h
    if sid == 1:
        return Scenario(
            sid=1,
            name="S1",
            pattern=[(9999.0, 60.0)],
            loop=True,
            route_type="straight",
            weather=None,  # 기본 날씨 유지
            target_kmh=60.0
        )
    # 시나리오 2: 직선, Heavy Rain, 60 km/h
    if sid == 2:
        heavy = carla.WeatherParameters(
            cloudiness=85.0, precipitation=90.0, precipitation_deposits=80.0,
            wetness=90.0, wind_intensity=0.4,
            sun_azimuth_angle=20.0, sun_altitude_angle=55.0,
            fog_density=8.0, fog_distance=0.0, fog_falloff=0.0
        )
        return Scenario(
            sid=2,
            name="S2",
            pattern=[(9999.0, 60.0)],
            loop=True,
            route_type="straight",
            weather=heavy,
            target_kmh=60.0
        )
    # 시나리오 3: 직선, WetNoon, 55 km/h
    if sid == 3:
        return Scenario(
            sid=3,
            name="S3",
            pattern=[(9999.0, 55.0)],
            loop=True,
            route_type="straight",
            weather=getattr(carla.WeatherParameters, "WetNoon"),
            target_kmh=55.0
        )
    # 기본은 시나리오 0
    return get_scenario(0, original_weather)

# =========================
# 날씨/마찰
# =========================
def apply_tire_friction(vehicle: carla.Vehicle, tire_mu: float):
    pc = vehicle.get_physics_control()
    for w in pc.wheels:
        w.tire_friction = tire_mu
    vehicle.apply_physics_control(pc)

# =========================
# Pure Pursuit 경로
# =========================
def build_forward_path(amap: carla.Map, start_tf: carla.Transform,
                       max_length_m=2000.0, step_m=2.0):
    wp = amap.get_waypoint(start_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    path=[wp]; total=0.0
    while total<max_length_m:
        nxt=path[-1].next(step_m)
        if not nxt: break
        path.append(nxt[0]); total+=step_m
    return path

def find_lookahead_target(path, cur_loc, lookahead_m, start_idx):
    i=clamp(start_idx,0,len(path)-1)
    best_i=i; best_d2=1e18
    for k in range(max(0,i-20), min(len(path),i+21)):
        d2=path[k].transform.location.distance(cur_loc)**2
        if d2<best_d2: best_d2=d2; best_i=k
    acc=0.0; idx=best_i; target_tf=path[idx].transform
    while idx<len(path)-1 and acc<lookahead_m:
        a=path[idx].transform.location; b=path[idx+1].transform.location
        acc+=a.distance(b); idx+=1; target_tf=path[idx].transform
    return idx, target_tf

# =========================
# 입력 스레드
# =========================
def start_input_thread(cmd_q: "queue.Queue[str]"):
    def _run():
        while True:
            s=sys.stdin.readline()
            if not s: continue
            s=s.strip()
            if s in ("0","1","2","3","q","Q"): cmd_q.put(s)
    threading.Thread(target=_run, daemon=True).start()

# =========================
# 스폰 유틸 (직선 시나리오: 53번 기준 200 m 뒤)
# =========================
def spawn_vehicle_at(world: carla.World, bp: carla.ActorBlueprint, tf: carla.Transform) -> carla.Vehicle:
    for dz in (0.0, 0.5, 1.0):
        tf_try=carla.Transform(
            carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z+dz),
            tf.rotation
        )
        v=world.try_spawn_actor(bp, tf_try)
        if v: return v
    raise RuntimeError("spawn 실패")

def compute_straight_spawn(amap: carla.Map, sps: List[carla.Transform], base_index: int, back_m: float) -> carla.Transform:
    tf0 = sps[min(max(0,base_index),len(sps)-1)]
    wp = amap.get_waypoint(tf0.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    # 차선 방향으로 back_m 만큼 뒤로 이동
    prevs = wp.previous(back_m)
    base_wp = prevs[0] if prevs else wp
    return base_wp.transform

def update_track_frame_for_straight(amap: carla.Map, start_tf: carla.Transform, length_m: float=700.0) -> None:
    """직선 시나리오용 s 프레임 재설정: 시작=start_tf, 끝=start에서 length_m 앞으로"""
    global track_frame
    # 앞으로 length_m 만큼 진척한 위치를 end로 사용
    wp = amap.get_waypoint(start_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    acc = 0.0
    cur = wp
    step = 2.0
    while acc < length_m:
        nxt = cur.next(step)
        if not nxt: break
        acc += cur.transform.location.distance(nxt[0].transform.location)
        cur = nxt[0]
    start_loc = start_tf.location
    end_loc   = cur.transform.location
    track_frame.set_line(start_loc, end_loc)

# =========================
# 메인
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--spawn_idx", type=int, default=20)  # 시나리오 0에서 사용
    ap.add_argument("--dry_mu", type=float, default=3.0)
    ap.add_argument("--wet_mu", type=float, default=1.6)  # (참고) 리드는 항상 dry 사용
    ap.add_argument("--start_scenario", type=int, default=0)  # 0~3
    ap.add_argument("--pp_gain", type=float, default=1.6)
    ap.add_argument("--lookahead_min", type=float, default=8.0)
    ap.add_argument("--lookahead_max", type=float, default=15.0)
    args=ap.parse_args()

    client=carla.Client(args.host,args.port); client.set_timeout(10.0)
    world=client.get_world(); amap=world.get_map()
    bp_lib=world.get_blueprint_library(); sps=amap.get_spawn_points()
    original_weather=world.get_weather()

    # 충돌 센서
    collision_sensor_bp = bp_lib.find('sensor.other.collision')

    # 월드 클린업(리드만 정리)
    for a in world.get_actors().filter("vehicle.*"):
        try:a.destroy()
        except:pass
    print("[CLEANUP] 모든 차량 제거 완료")

    def spawn_lead_for_scenario(scn: Scenario) -> carla.Vehicle:
        lead_bp=(bp_lib.filter('vehicle.audi.tt') or bp_lib.filter('vehicle.*'))[0]
        lead_bp.set_attribute('role_name','lead')

        if scn.route_type == "mixed":
            tf = sps[min(max(0, args.spawn_idx), len(sps)-1)]
            wp=amap.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            base_tf=wp.transform
            v = spawn_vehicle_at(world, lead_bp, base_tf)
            # s 프레임: 기존 DEFAULT 그대로
            track_frame.set_line(DEFAULT_LINE_START, DEFAULT_LINE_END)
            return v
        else:
            # straight: 스폰포인트 53에서 차선방향 기준 200 m 뒤
            tf_st = compute_straight_spawn(amap, sps, base_index=53, back_m=200.0)
            v = spawn_vehicle_at(world, lead_bp, tf_st)
            # s 프레임 갱신 (700 m 직선 기준)
            update_track_frame_for_straight(amap, tf_st, length_m=700.0)
            return v

    # 리드 차량 스폰
    sid=int(args.start_scenario); scenario=get_scenario(sid, original_weather)

    lead=spawn_lead_for_scenario(scenario)
    apply_tire_friction(lead, args.dry_mu)   # 리드는 항상 dry_mu 유지

    # 충돌 센서 부착
    collided_flag = {"hit": False}
    def on_collision(event):
        other = event.other_actor
        print(f"[COLLISION] 리드와 충돌: other={other.type_id}")
        collided_flag["hit"] = True
    collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=lead)
    collision_sensor.listen(on_collision)

    # 날씨 적용(시나리오별 단일 날씨)
    if scenario.weather is None:
        world.set_weather(original_weather)
    else:
        world.set_weather(scenario.weather)

    path=build_forward_path(amap, lead.get_transform())
    speed_pid=PID(0.60, 0.05, 0.02)
    steer_gain=args.pp_gain
    lookahead_min=args.lookahead_min; lookahead_max=args.lookahead_max
    idx_hint=0

    seg_i, seg_elapsed=0, 0.0
    print(f"[SCENARIO] {scenario.name}  (0/1/2/3 전환, q 종료)")

    prev_kmh_cmd=scenario.pattern[0][1]

    # 출발/비상 킥 (정지→출발 보조)
    KICK_DURATION=1.0; KICK_THROTTLE=0.55; kick_until_ts=0.0
    STUCK_SPEED_MS=0.5; STUCK_TIMEOUT=2.0
    EMERG_KICK_DURATION=1.2; emerg_kick_until_ts=0.0
    stuck_since=None

    cmd_q: "queue.Queue[str]" = queue.Queue()
    start_input_thread(cmd_q)

    last_dt=0.05
    try:
        while True:
            try: snap=world.wait_for_tick()
            except KeyboardInterrupt: break
            dt = snap.timestamp.delta_seconds if snap else last_dt; last_dt=dt
            now=time.time()

            # 입력 처리
            try: cmd=cmd_q.get_nowait()
            except queue.Empty: cmd=None
            if cmd in ("q","Q"): break

            # 시나리오 전환
            if cmd in ("0","1","2","3"):
                # 리드 및 센서 정리
                try: collision_sensor.stop()
                except: pass
                for a in world.get_actors().filter("vehicle.*"):
                    try:a.destroy()
                    except:pass
                # 날씨 복구 후, 새 시나리오 날씨 적용
                world.set_weather(original_weather)

                sid=int(cmd); scenario=get_scenario(sid, original_weather)
                lead=spawn_lead_for_scenario(scenario)
                apply_tire_friction(lead, args.dry_mu)
                path=build_forward_path(amap, lead.get_transform())
                speed_pid.reset(); idx_hint=0
                seg_i, seg_elapsed=0, 0.0
                prev_kmh_cmd=scenario.pattern[0][1]
                kick_until_ts=0.0; emerg_kick_until_ts=0.0; stuck_since=None
                collided_flag["hit"]=False

                # 충돌 센서 재부착
                collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=lead)
                collision_sensor.listen(on_collision)

                # 시나리오별 날씨 재적용
                if scenario.weather is None:
                    world.set_weather(original_weather)
                else:
                    world.set_weather(scenario.weather)

                print(f"\n[SCENARIO] {scenario.name}")
                continue

            # 충돌 시 즉시 정지 유지
            if collided_flag["hit"]:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False))
                speed_pid.reset()
                continue

            tf_cur=lead.get_transform(); loc=tf_cur.location; yaw=tf_cur.rotation.yaw
            vv=lead.get_velocity(); speed=math.sqrt(vv.x*vv.x + vv.y*vv.y + vv.z*vv.z); speed_kmh=3.6*speed
            if idx_hint>len(path)-50: path=build_forward_path(amap, tf_cur); idx_hint=0

            # 진행도 s
            s_on_line = track_frame.s_on_line(loc)

            # +700m: 정지 (직선 시나리오/혼합 루트 공통)
            if s_on_line >= STOP_S:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                speed_pid.reset()
                # 멈춘 상태 유지
                continue

            # 시나리오 진행(여기서는 단일 속도 정책)
            duration, kmh_cmd = scenario.pattern[seg_i]
            if seg_elapsed >= duration:
                seg_i += 1; seg_elapsed = 0.0
                if seg_i >= len(scenario.pattern):
                    if scenario.loop: seg_i = 0
                    else: seg_i=len(scenario.pattern)-1
                duration, kmh_cmd = scenario.pattern[seg_i]
            seg_elapsed += dt

            # 650~700 m 선형 감속
            if DECEL_START_S <= s_on_line < STOP_S:
                v_allow = MAX_CRUISE_KMH * (STOP_S - s_on_line) / (STOP_S - DECEL_START_S)
                kmh_cmd = min(kmh_cmd, v_allow)

            # 커브 감속(시나리오 0에서 효과적, 직선 시나리오는 영향 거의 없음)
            try:
                wp_now = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                next_wps = wp_now.next(25.0)
                if next_wps:
                    yaw_next = next_wps[0].transform.rotation.yaw
                    yaw_diff = abs(deg_wrap180(yaw_next - wp_now.transform.rotation.yaw))
                    if yaw_diff > 25: kmh_cmd = min(kmh_cmd, 30.0)
                    elif yaw_diff > 12: kmh_cmd = min(kmh_cmd, 40.0)
            except: pass

            # Pure Pursuit
            lookahead = clamp(lookahead_min + (lookahead_max - lookahead_min)*(speed_kmh/80.0),
                              lookahead_min, lookahead_max)
            idx_hint, tf_tgt = find_lookahead_target(path, loc, lookahead, idx_hint)
            tgt = tf_tgt.location
            dx,dy = tgt.x - loc.x, tgt.y - loc.y
            yaw_rad = math.radians(yaw); fx,fy = math.cos(yaw_rad), math.sin(yaw_rad)
            y_left = -dx*fy + dy*fx
            Ld = max(3.0, lookahead)
            steer = clamp(steer_gain * (2.0 * y_left) / (Ld*Ld), -1.0, 1.0)

            # 속도 제어 + 출발/비상 킥
            v_ref = kmh_cmd / 3.6
            a_cmd = speed_pid.step(v_ref - speed, dt)
            throttle=0.0; brake=0.0

            # 출발 킥 조건(정지→가속)
            if ((prev_kmh_cmd <= 0.1) or (speed < STUCK_SPEED_MS)) and (kmh_cmd > 0.1):
                if now >= kick_until_ts: kick_until_ts = now + KICK_DURATION

            # 스턱 감시 → 비상 킥
            if kmh_cmd > 0.1:
                if speed < STUCK_SPEED_MS:
                    if stuck_since is None: stuck_since = now
                    elif (now - stuck_since) >= STUCK_TIMEOUT:
                        emerg_kick_until_ts = now + EMERG_KICK_DURATION
                        stuck_since = None
                else: stuck_since=None
            else: stuck_since=None

            if now < emerg_kick_until_ts:
                throttle = max(throttle, 0.60); brake=0.0
            elif now < kick_until_ts:
                throttle = max(throttle, KICK_THROTTLE); brake=0.0
            else:
                if kmh_cmd <= 0.1:
                    throttle=0.0; brake=1.0; speed_pid.reset()
                else:
                    if a_cmd >= 0: throttle = clamp(a_cmd, 0.0, 0.9)
                    else: brake = clamp(-a_cmd, 0.0, 1.0)

            prev_kmh_cmd=kmh_cmd
            lead.apply_control(carla.VehicleControl(throttle=float(throttle),
                                                    steer=float(steer),
                                                    brake=float(brake)))
    finally:
        try:
            world.set_weather(original_weather)
        except: pass
        for a in world.get_actors().filter("vehicle.*"):
            try:a.destroy()
            except:pass
        print("[CLEANUP] 리드 제거 & 날씨 복구 완료")

if __name__ == "__main__":
    main()