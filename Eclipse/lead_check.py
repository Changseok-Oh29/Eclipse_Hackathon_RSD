#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lead_check.py — Pure Pursuit + 시나리오 속도제어 (리드 차량은 날씨만 변경, μ는 고정)

요구사항 반영:
- 1/2/3/4 공통 로직: 커브 자동 감속, 650~700 m 선형 감속, 700 m 정지
- 시나리오 전환(1/2/3/4) 또는 Ctrl+C 종료 시: 날씨를 '원래 날씨'로 복구 + 타이어 마찰도 dry로 복구
- +300 m에서 Heavy Rain으로 날씨만 바뀜 (리드 차량 μ는 dry 유지 → 끊김 없이 주행)
- S4에서 '정지→재출발'이 항상 일어나도록:
  * 0→가속 전환 시 출발 킥(기본 1.0 s, 스로틀 0.55)
  * kmh_cmd>0 인데도 2초 이상 0.5 m/s 미만이면 비상 킥(1.2 s) 발동
- TM/Autopilot/agents 모듈 사용 안 함. spawn_idx=20 고정(충돌 시 z offset 재시도).
"""

import sys, time, math, threading, queue
from dataclasses import dataclass
from typing import List, Tuple
import carla

# =========================
# 유틸
# =========================
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def deg_wrap180(a): return (a + 180.0) % 360.0 - 180.0
def norm2d(x, y): return math.sqrt(x*x + y*y)
def dot2d(ax, ay, bx, by): return ax*bx + ay*by
def _cm(x): return x/100.0

# =========================
# 두 번째 직선 기준
# =========================
SECOND_STRAIGHT_START = carla.Location(x=_cm(34340),  y=_cm(2169),  z=_cm(148))
SECOND_STRAIGHT_END   = carla.Location(x=_cm(-36211), y=_cm(1580),  z=_cm(165))

RAIN_SWITCH_S  = 300.0   # +300 m : 비 전환
DECEL_START_S  = 650.0   # +650 m : 감속 시작
STOP_S         = 700.0   # +700 m : 정지
MAX_CRUISE_KMH = 60.0    # 선형 감속 상한 기준 속도

def along_track_s_on_second_straight(p: carla.Location) -> float:
    sx, sy = SECOND_STRAIGHT_START.x, SECOND_STRAIGHT_START.y
    ex, ey = SECOND_STRAIGHT_END.x,   SECOND_STRAIGHT_END.y
    dx, dy = (ex - sx), (ey - sy)
    L = norm2d(dx, dy)
    if L < 1e-3: return -1e9
    ux, uy = dx/L, dy/L
    px, py = p.x - sx, p.y - sy
    return dot2d(px, py, ux, uy)

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
    name: str
    pattern: List[Tuple[float, float]]
    loop: bool

def get_scenario(sid: int) -> Scenario:
    if sid == 1:
        return Scenario("S1", [(8.0,30.0),(8.0,45.0),(8.0,60.0),(6.0,35.0),(10.0,60.0)], False)
    if sid == 2:
        return Scenario("S2", [(5.0,40.0),(5.0,60.0)], True)
    if sid == 3:
        return Scenario("S3", [(8.0,60.0),(2.0,0.0)], True)
    if sid == 4:
        return Scenario("S4", [(4.0,0.0),(6.0,60.0),(6.0,40.0),(6.0,60.0),(4.0,0.0),
                               (6.0,60.0),(6.0,40.0),(6.0,60.0),(4.0,0.0)], False)
    return get_scenario(1)

# =========================
# 날씨
# =========================
@dataclass
class WeatherProfile:
    name: str
    params: carla.WeatherParameters

def weather_heavy_rain_day() -> WeatherProfile:
    return WeatherProfile(
        name="heavy_rain_day",
        params=carla.WeatherParameters(
            cloudiness=85.0, precipitation=90.0, precipitation_deposits=80.0,
            wetness=90.0, wind_intensity=0.4,
            sun_azimuth_angle=20.0, sun_altitude_angle=55.0,
            fog_density=8.0, fog_distance=0.0, fog_falloff=0.0
        )
    )

def apply_tire_friction(vehicle: carla.Vehicle, tire_mu: float):
    pc = vehicle.get_physics_control()
    for w in pc.wheels:
        w.tire_friction = tire_mu
    vehicle.apply_physics_control(pc)

# =========================
# Pure Pursuit
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
            if s in ("1","2","3","4","q","Q"): cmd_q.put(s)
    threading.Thread(target=_run, daemon=True).start()

# =========================
# 메인
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--spawn_idx", type=int, default=20)
    ap.add_argument("--dry_mu", type=float, default=3.0)
    ap.add_argument("--wet_mu", type=float, default=1.6)
    ap.add_argument("--start_scenario", type=int, default=1)
    ap.add_argument("--pp_gain", type=float, default=1.6)
    ap.add_argument("--lookahead_min", type=float, default=8.0)
    ap.add_argument("--lookahead_max", type=float, default=15.0)
    args=ap.parse_args()

    client=carla.Client(args.host,args.port); client.set_timeout(10.0)
    world=client.get_world(); amap=world.get_map()
    bp=world.get_blueprint_library(); sps=amap.get_spawn_points()
    original_weather=world.get_weather()

    for a in world.get_actors().filter("vehicle.*"):
        try:a.destroy()
        except:pass
    print("[CLEANUP] 모든 차량 제거 완료")

    def spawn_lead(idx:int)->carla.Vehicle:
        tf=sps[min(max(0,idx),len(sps)-1)]
        wp=amap.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        base_tf=wp.transform
        lead_bp=(bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
        lead_bp.set_attribute('role_name','lead')
        for dz in (0.0, 0.5, 1.0):
            tf_try=carla.Transform(
                carla.Location(x=base_tf.location.x, y=base_tf.location.y, z=base_tf.location.z+dz),
                base_tf.rotation
            )
            v=world.try_spawn_actor(lead_bp, tf_try)
            if v: return v
        raise RuntimeError("spawn 실패")

    lead=spawn_lead(args.spawn_idx)
    apply_tire_friction(lead, args.dry_mu)   # 리드는 항상 dry_mu 유지

    path=build_forward_path(amap, lead.get_transform())
    speed_pid=PID(0.60, 0.05, 0.02)
    steer_gain=args.pp_gain
    lookahead_min=args.lookahead_min; lookahead_max=args.lookahead_max
    idx_hint=0

    sid=int(args.start_scenario); scenario=get_scenario(sid)
    seg_i, seg_elapsed=0, 0.0
    print(f"[SCENARIO] {scenario.name}  (1/2/3/4 전환, q 종료)")

    weather_switched=False
    prev_kmh_cmd=scenario.pattern[0][1]

    # 출발/비상 킥
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

            try: cmd=cmd_q.get_nowait()
            except queue.Empty: cmd=None
            if cmd in ("q","Q"): break

            if cmd in ("1","2","3","4"):
                for a in world.get_actors().filter("vehicle.*"):
                    try:a.destroy()
                    except:pass
                world.set_weather(original_weather)
                lead=spawn_lead(args.spawn_idx)
                apply_tire_friction(lead, args.dry_mu)
                path=build_forward_path(amap, lead.get_transform())
                speed_pid.reset(); idx_hint=0
                sid=int(cmd); scenario=get_scenario(sid)
                seg_i, seg_elapsed=0, 0.0
                weather_switched=False
                prev_kmh_cmd=scenario.pattern[0][1]
                kick_until_ts=0.0; emerg_kick_until_ts=0.0; stuck_since=None
                print(f"\n[SCENARIO] {scenario.name}")
                continue

            tf_cur=lead.get_transform(); loc=tf_cur.location; yaw=tf_cur.rotation.yaw
            vv=lead.get_velocity(); speed=math.sqrt(vv.x*vv.x + vv.y*vv.y + vv.z*vv.z); speed_kmh=3.6*speed
            if idx_hint>len(path)-50: path=build_forward_path(amap, tf_cur); idx_hint=0
            s_on_line=along_track_s_on_second_straight(loc)

            # +300m: 날씨만 변경 
            if (not weather_switched) and (s_on_line>=RAIN_SWITCH_S):
                world.set_weather(weather_heavy_rain_day().params)
                weather_switched=True
                print("[WEATHER] +300m → Heavy Rain")

            # +700m: 정지
            if s_on_line >= STOP_S:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                speed_pid.reset()
                print("vehice Stopped")
                continue

            # 시나리오 진행
            duration, kmh_cmd = scenario.pattern[seg_i]
            if seg_elapsed >= duration:
                seg_i += 1; seg_elapsed = 0.0
                if seg_i >= len(scenario.pattern):
                    if scenario.loop: seg_i = 0
                    else:
                        if scenario.name=="S4" and s_on_line<DECEL_START_S: seg_i=0
                        else: seg_i=len(scenario.pattern)-1
                duration, kmh_cmd = scenario.pattern[seg_i]
            seg_elapsed += dt

            if DECEL_START_S <= s_on_line < STOP_S:
                v_allow = MAX_CRUISE_KMH * (STOP_S - s_on_line) / (STOP_S - DECEL_START_S)
                kmh_cmd = min(kmh_cmd, v_allow)

            try:
                wp_now = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                next_wps = wp_now.next(25.0)
                if next_wps:
                    yaw_next = next_wps[0].transform.rotation.yaw
                    yaw_diff = abs(deg_wrap180(yaw_next - wp_now.transform.rotation.yaw))
                    if yaw_diff > 25: kmh_cmd = min(kmh_cmd, 30.0)
                    elif yaw_diff > 12: kmh_cmd = min(kmh_cmd, 40.0)
            except: pass

            lookahead = clamp(lookahead_min + (lookahead_max - lookahead_min)*(speed_kmh/80.0),
                              lookahead_min, lookahead_max)
            idx_hint, tf_tgt = find_lookahead_target(path, loc, lookahead, idx_hint)
            tgt = tf_tgt.location
            dx,dy = tgt.x - loc.x, tgt.y - loc.y
            yaw_rad = math.radians(yaw); fx,fy = math.cos(yaw_rad), math.sin(yaw_rad)
            y_left = -dx*fy + dy*fx
            Ld = max(3.0, lookahead)
            steer = clamp(steer_gain * (2.0 * y_left) / (Ld*Ld), -1.0, 1.0)

            v_ref = kmh_cmd / 3.6
            a_cmd = speed_pid.step(v_ref - speed, dt)
            throttle=0.0; brake=0.0

            if ((prev_kmh_cmd <= 0.1) or (speed < STUCK_SPEED_MS)) and (kmh_cmd > 0.1):
                if now >= kick_until_ts: kick_until_ts = now + KICK_DURATION

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
        world.set_weather(original_weather)
        for a in world.get_actors().filter("vehicle.*"):
            try:a.destroy()
            except:pass
        print("[CLEANUP] 리드 제거 & 날씨 복구 완료")

if __name__ == "__main__":
    main()

