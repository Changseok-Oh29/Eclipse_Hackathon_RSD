#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lead_scenario.py — Pure Pursuit + 시나리오 속도제어 (env.py에서 만든 ego/lead 제어)

요약:
- env.py에서 ego/lead 차량 스폰 → 이 파일은 ego/lead를 찾아와 제어
- 시나리오 0: ego=spawn idx 20, lead=앞쪽 30 m
- 시나리오 1~3: ego=spawn idx 53, lead=앞쪽 30 m
- 날씨는 시나리오별로 변경
- 리드 차량 μ 항상 dry 유지
- 키보드: 0/1/2/3 전환, q 종료
"""

import sys, time, math, threading, queue
from dataclasses import dataclass, field
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
# s-프레임 정의
# =========================
class TrackFrame:
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

DEFAULT_LINE_START = carla.Location(x=343.40,  y=21.69,  z=1.48)
DEFAULT_LINE_END   = carla.Location(x=-362.11, y=15.80,  z=1.65)
track_frame = TrackFrame(DEFAULT_LINE_START, DEFAULT_LINE_END)

MAX_CRUISE_KMH = 60.0

# =========================
# PID
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
# 시나리오 정의
# =========================
@dataclass
class Scenario:
    sid: int
    name: str
    pattern: List[Tuple[float, float]]
    loop: bool
    weather: Optional[carla.WeatherParameters]
    target_kmh: float
    decel_start: float
    stop_s: float
    stop_zones: List[float] = field(default_factory=list)

def get_scenario(sid: int, original_weather: carla.WeatherParameters) -> Scenario:
    if sid == 0:
        return Scenario(0, "S0", [(9999.0, 60.0)], True,
                        None, 60.0, 650.0, 700.0)
    if sid == 1:
        return Scenario(1, "S1", [(9999.0, 60.0)], True,
                        None, 60.0, 550.0, 600.0)
    if sid == 2:
        heavy = carla.WeatherParameters(
            cloudiness=85.0, precipitation=90.0, precipitation_deposits=80.0,
            wetness=90.0, wind_intensity=0.4,
            sun_azimuth_angle=20.0, sun_altitude_angle=55.0,
            fog_density=8.0, fog_distance=0.0, fog_falloff=0.0
        )
        return Scenario(2, "S2", [(9999.0, 60.0)], True,
                        heavy, 60.0, 550.0, 600.0)
    if sid == 3:
        return Scenario(3, "S3", [(9999.0, 55.0)], True,
                        getattr(carla.WeatherParameters, "WetNoon"),
                        55.0, 550.0, 600.0, stop_zones=[200.0, 400.0])
    return get_scenario(0, original_weather)

# =========================
# 물리
# =========================
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
    wp = amap.get_waypoint(start_tf.location, project_to_road=True,
                           lane_type=carla.LaneType.Driving)
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
# 차량 찾기
# =========================
def find_ego(world: carla.World):
    egos = [a for a in world.get_actors().filter("vehicle.*")
            if a.attributes.get("role_name")=="ego"]
    if not egos:
        raise RuntimeError("No ego vehicle found. Run env.py first.")
    return egos[0]

def find_lead(world: carla.World):
    leads = [a for a in world.get_actors().filter("vehicle.*")
             if a.attributes.get("role_name")=="lead"]
    if not leads:
        raise RuntimeError("No lead vehicle found. Run env.py first.")
    return leads[0]

# =========================
# 위치 리셋
# =========================
def reset_positions(world, amap, ego, lead, scenario: Scenario, sps: List[carla.Transform]):
    if scenario.sid == 0:
        base_tf = sps[20]
    else:
        base_tf = sps[53]

    # ego 위치 리셋
    ego.set_simulate_physics(False)
    ego.set_transform(base_tf)
    ego.set_simulate_physics(True)

    # lead = ego 앞쪽 30m
    wp = amap.get_waypoint(base_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    forward = wp.next(30.0)
    if forward:
        lead_tf = forward[0].transform
        lead_tf.location.z = base_tf.location.z
        lead.set_simulate_physics(False)
        lead.set_transform(lead_tf)
        lead.set_simulate_physics(True)

# =========================
# 메인
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--dry_mu", type=float, default=3.0)
    ap.add_argument("--start_scenario", type=int, default=0)
    ap.add_argument("--pp_gain", type=float, default=1.6)
    ap.add_argument("--lookahead_min", type=float, default=8.0)
    ap.add_argument("--lookahead_max", type=float, default=15.0)
    args=ap.parse_args()

    client=carla.Client(args.host,args.port); client.set_timeout(10.0)
    world=client.get_world(); amap=world.get_map()
    sps=amap.get_spawn_points()
    original_weather=world.get_weather()

    ego=find_ego(world); lead=find_lead(world)
    apply_tire_friction(lead, args.dry_mu)

    sid=int(args.start_scenario); scenario=get_scenario(sid, original_weather)
    reset_positions(world, amap, ego, lead, scenario, sps)
    world.set_weather(original_weather if scenario.weather is None else scenario.weather)

    path=build_forward_path(amap, lead.get_transform())
    speed_pid=PID(0.60, 0.05, 0.02)
    steer_gain=args.pp_gain
    lookahead_min=args.lookahead_min; lookahead_max=args.lookahead_max
    idx_hint=0

    seg_i, seg_elapsed=0, 0.0
    print(f"[SCENARIO] {scenario.name}  (0/1/2/3 전환, q 종료)")

    prev_kmh_cmd=scenario.pattern[0][1]

    KICK_DURATION=1.0; KICK_THROTTLE=0.55; kick_until_ts=0.0
    STUCK_SPEED_MS=0.5; STUCK_TIMEOUT=2.0
    EMERG_KICK_DURATION=1.2; emerg_kick_until_ts=0.0
    stuck_since=None
    stop_state = {"active": False, "resume_ts": 0.0, "visited": set()}

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

            if cmd in ("0","1","2","3"):
                speed_pid.reset(); idx_hint=0
                sid=int(cmd); scenario=get_scenario(sid, original_weather)
                reset_positions(world, amap, ego, lead, scenario, sps)
                seg_i, seg_elapsed=0, 0.0
                prev_kmh_cmd=scenario.pattern[0][1]
                kick_until_ts=0.0; emerg_kick_until_ts=0.0; stuck_since=None
                stop_state = {"active": False, "resume_ts": 0.0, "visited": set()}
                world.set_weather(original_weather if scenario.weather is None else scenario.weather)
                print(f"\n[SCENARIO] {scenario.name}")
                continue

            # ===== 제어 루프 (lead) =====
            tf_cur=lead.get_transform(); loc=tf_cur.location; yaw=tf_cur.rotation.yaw
            vv=lead.get_velocity(); speed=math.sqrt(vv.x*vv.x + vv.y*vv.y + vv.z*vv.z); speed_kmh=3.6*speed
            if idx_hint>len(path)-50: path=build_forward_path(amap, tf_cur); idx_hint=0
            s_on_line = track_frame.s_on_line(loc)

            # stop_zones
            if stop_state["active"]:
                if now >= stop_state["resume_ts"]:
                    stop_state["active"] = False
                    speed_pid.reset()
                else:
                    lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                    continue
            else:
                for sz in scenario.stop_zones:
                    if sz in stop_state["visited"]: continue
                    if abs(s_on_line - sz) <= 5.0:
                        print(f"[EVENT] Stop zone at {sz:.1f} m → full stop 5s")
                        stop_state["visited"].add(sz)
                        stop_state["active"] = True
                        stop_state["resume_ts"] = now + 5.0
                        lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                        speed_pid.reset()
                        continue

            # 최종 정지
            if s_on_line >= scenario.stop_s:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                speed_pid.reset()
                continue

            # 목표 속도
            duration, kmh_cmd = scenario.pattern[seg_i]
            if seg_elapsed >= duration:
                seg_i += 1; seg_elapsed = 0.0
                if seg_i >= len(scenario.pattern):
                    seg_i = 0 if scenario.loop else len(scenario.pattern)-1
                duration, kmh_cmd = scenario.pattern[seg_i]
            seg_elapsed += dt

            if scenario.decel_start <= s_on_line < scenario.stop_s:
                v_allow = MAX_CRUISE_KMH * (scenario.stop_s - s_on_line) / max((scenario.stop_s - scenario.decel_start), 1e-3)
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

            # Pure Pursuit 조향
            lookahead = clamp(lookahead_min + (lookahead_max - lookahead_min)*(speed_kmh/80.0),
                              lookahead_min, lookahead_max)
            idx_hint, tf_tgt = find_lookahead_target(path, loc, lookahead, idx_hint)
            tgt = tf_tgt.location
            dx,dy = tgt.x - loc.x, tgt.y - loc.y
            yaw_rad = math.radians(yaw); fx,fy = math.cos(yaw_rad), math.sin(yaw_rad)
            y_left = -dx*fy + dy*fx
            Ld = max(3.0, lookahead)
            steer = clamp(steer_gain * (2.0 * y_left) / (Ld*Ld), -1.0, 1.0)

            # 속도 제어
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
        print("[CLEANUP] 날씨 복구 완료")

if __name__ == "__main__":
    main()
