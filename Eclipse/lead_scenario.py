#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lead_scenario.py — env.py가 스폰한 ego/lead를 찾아 제어 (안정 주행 개편판)

핵심 변경
- Pure Pursuit → Stanley 컨트롤러(차선 중심 고정)
- 동일 차선 경로(road_id/lane_id 고정) 생성
- 조향 속도 제한 + 저역통과로 핸들 진동 억제
- 곡률 기반 추가 감속

키: 0/1/2/3 전환, q 종료
"""

import sys, time, math, threading, queue
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import carla

# =========================
# 유틸/보조
# =========================
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def deg_wrap180(a): return (a + 180.0) % 360.0 - 180.0
def norm2d(x, y): return math.sqrt(x*x + y*y)
def dot2d(ax, ay, bx, by): return ax*bx + ay*by

def zero_vel(v: carla.Vehicle):
    try:
        v.set_velocity(carla.Vector3D(0,0,0))
        v.set_angular_velocity(carla.Vector3D(0,0,0))
    except: pass

def teleport(v: carla.Vehicle, tf: carla.Transform):
    v.set_simulate_physics(False)
    v.set_transform(tf)
    zero_vel(v)
    v.set_simulate_physics(True)

# =========================
# s-프레임 정의 (정지구간 계산용)
# =========================
class TrackFrame:
    def __init__(self, start: carla.Location, end: carla.Location):
        self.set_line(start, end)
    def set_line(self, start: carla.Location, end: carla.Location):
        self.sx, self.sy = start.x, start.y
        self.ex, self.ey = end.x, end.y
        dx, dy = (self.ex - self.sx), (self.ey - self.sy)
        self.L = norm2d(dx, dy)
        self.ux, self.uy = (1.0,0.0) if self.L<1e-6 else (dx/self.L, dy/self.L)
    def s_on_line(self, p: carla.Location) -> float:
        if self.L < 1e-6: return -1e9
        px, py = p.x - self.sx, p.y - self.sy
        return dot2d(px, py, self.ux, self.uy)

DEFAULT_LINE_START = carla.Location(x=343.40,  y=21.69,  z=1.48)
DEFAULT_LINE_END   = carla.Location(x=-362.11, y=15.80,  z=1.65)
track_frame = TrackFrame(DEFAULT_LINE_START, DEFAULT_LINE_END)

MAX_CRUISE_KMH = 60.0

def update_track_frame_for_straight(amap: carla.Map, start_tf: carla.Transform, length_m: float=750.0):
    """직선 시나리오에서 s-프레임을 시드 위치부터 length_m 앞으로 정의"""
    global track_frame
    wp = amap.get_waypoint(start_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    acc=0.0; cur=wp
    while acc < length_m:
        nxt = cur.next(2.0)
        if not nxt: break
        acc += cur.transform.location.distance(nxt[0].transform.location)
        cur = nxt[0]
    track_frame.set_line(start_tf.location, cur.transform.location)

# =========================
# PID (가감속용)
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
# 동일-차선 경로 생성 + 최근접/전방 포인트
# =========================
def build_same_lane_path(amap: carla.Map, start_wp: carla.Waypoint,
                         max_length_m=2000.0, step_m=2.0):
    """start_wp의 road_id/lane_id를 유지하며 앞으로 진행하는 경로"""
    road_id = start_wp.road_id
    lane_id = start_wp.lane_id
    path=[start_wp]; total=0.0; cur=start_wp
    while total < max_length_m:
        nxts = cur.next(step_m)
        if not nxts: break
        nxt = nxts[0]
        # 동일 차선 유지: 분기/교차로에서도 lane/road 고정(가능한 한)
        if (nxt.road_id != road_id) or (nxt.lane_id != lane_id):
            # 같은 road로 이어지지 않으면 가장 가까운 동일-차선 후보 탐색
            cand = [w for w in nxts if (w.road_id==road_id and w.lane_id==lane_id)]
            if cand:
                nxt = cand[0]
            else:
                # 동일 차선 불가 → 현재 차선의 다음으로 계속 (차선변경 최소화)
                pass
        path.append(nxt)
        total += step_m
        cur = nxt
    return path

def nearest_index_on_path(path, loc, hint=0, search_window=40):
    i = clamp(hint, 0, len(path)-1)
    lo = max(0, i - search_window)
    hi = min(len(path)-1, i + search_window)
    best_i = i; best_d2 = 1e18
    for k in range(lo, hi+1):
        p = path[k].transform.location
        d2 = (p.x-loc.x)**2 + (p.y-loc.y)**2
        if d2 < best_d2: best_d2=d2; best_i=k
    return best_i

# =========================
# Stanley 조향
# =========================
class StanleyController:
    def __init__(self, k_e=0.7, k_soft=1.0, steer_rate=1.5, alpha=0.25):
        """
        k_e: 횡오차 게인
        k_soft: 저속 안정화용 소프트닝
        steer_rate: 스티어링 속도 제한(단위: 1.0/초)
        alpha: 1차 저역통과 필터 계수(0~1, 높을수록 부드러움)
        """
        self.k_e = k_e
        self.k_soft = k_soft
        self.steer_rate = steer_rate
        self.alpha = alpha
        self.prev_steer = 0.0
        self.inited = False

    def reset(self):
        self.prev_steer = 0.0
        self.inited = False

    def step(self, path, idx, loc: carla.Location, yaw_deg: float, speed: float, dt: float):
        # 경로의 기준(현재 인덱스)과 그 법선/접선
        idx = clamp(idx, 0, len(path)-2)
        wp_a = path[idx].transform
        wp_b = path[idx+1].transform

        # 경로 heading
        hx, hy = (wp_b.location.x - wp_a.location.x), (wp_b.location.y - wp_a.location.y)
        hd = math.degrees(math.atan2(hy, hx))
        heading_err = deg_wrap180(hd - yaw_deg)
        heading_err_rad = math.radians(heading_err)

        # 횡오차(왼쪽 +)
        dx, dy = (loc.x - wp_a.location.x), (loc.y - wp_a.location.y)
        path_yaw = math.atan2(hy, hx)
        cross_track = -math.sin(path_yaw)*dx + math.cos(path_yaw)*dy  # 좌측(+)

        # Stanley 공식
        steer_cmd = heading_err_rad + math.atan2(self.k_e * cross_track, self.k_soft + speed)

        # 조향 속도 제한
        max_step = self.steer_rate * dt
        steer_cmd = clamp(steer_cmd, -1.2, 1.2)  # 여유 범위
        # 저역통과 + rate limit
        if not self.inited:
            smoothed = steer_cmd
            self.inited = True
        else:
            # 1차 LPF
            smoothed = (1.0 - self.alpha) * steer_cmd + self.alpha * self.prev_steer
            # rate limit
            delta = clamp(smoothed - self.prev_steer, -max_step, max_step)
            smoothed = self.prev_steer + delta

        self.prev_steer = smoothed
        return float(clamp(smoothed, -1.0, 1.0))

# =========================
# 시나리오
# =========================
@dataclass
class Scenario:
    sid: int
    name: str
    route_type: str  # "mixed" | "straight"
    weather: Optional[carla.WeatherParameters]
    target_kmh: float
    decel_start: float
    stop_s: float
    stop_zones: List[float] = field(default_factory=list)

def get_scenario(sid: int, original_weather: carla.WeatherParameters) -> Scenario:
    if sid == 0:
        return Scenario(0, "S0", "mixed", None, 60.0, 650.0, 700.0, [])
    if sid == 1:
        return Scenario(1, "S1", "straight", None, 60.0, 700.0, 750.0, [])
    if sid == 2:
        heavy = carla.WeatherParameters(
            cloudiness=85.0, precipitation=90.0, precipitation_deposits=80.0,
            wetness=90.0, wind_intensity=0.4,
            sun_azimuth_angle=20.0, sun_altitude_angle=55.0,
            fog_density=8.0, fog_distance=0.0, fog_falloff=0.0
        )
        return Scenario(2, "S2", "straight", heavy, 60.0, 700.0, 750.0, [])
    if sid == 3:
        return Scenario(3, "S3", "straight", getattr(carla.WeatherParameters, "WetNoon"),
                        55.0, 700.0, 750.0, [200.0, 400.0])
    return get_scenario(0, original_weather)

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
    xs = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name")=="ego"]
    if not xs: raise RuntimeError("No ego vehicle found. Run env.py first.")
    return xs[0]

def find_lead(world: carla.World):
    xs = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name")=="lead"]
    if not xs: raise RuntimeError("No lead vehicle found. Run env.py first.")
    return xs[0]

# =========================
# 위치 리셋 (ego/lead 동시 이동)
# =========================
def reset_positions(world, amap, ego, lead, scenario: Scenario, sps: List[carla.Transform]):
    if scenario.sid == 0:
        base_tf = sps[20]
        track_frame.set_line(DEFAULT_LINE_START, DEFAULT_LINE_END)
    else:
        tf53 = sps[53]
        wp = amap.get_waypoint(tf53.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        prevs = wp.previous(250.0)
        base_tf = (prevs[0] if prevs else wp).transform
        update_track_frame_for_straight(amap, base_tf, length_m=750.0)

    # ego (z+1)
    ego_tf = carla.Transform(
        carla.Location(base_tf.location.x, base_tf.location.y, base_tf.location.z+1.0),
        base_tf.rotation
    )
    teleport(ego, ego_tf)

    # lead = ego 앞 30m (z+1)
    wp_e = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    fw = wp_e.next(30.0)
    if fw:
        lead_tf = fw[0].transform
        lead_tf.location.z = ego_tf.location.z + 1.0
        teleport(lead, lead_tf)

# =========================
# 메인
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--start_scenario", type=int, default=0)
    # 속도/경로 튜닝
    ap.add_argument("--lookahead_for_curv", type=float, default=25.0)
    args=ap.parse_args()

    client=carla.Client(args.host,args.port); client.set_timeout(10.0)
    world=client.get_world(); amap=world.get_map()
    sps=amap.get_spawn_points(); original_weather=world.get_weather()
    bp_lib=world.get_blueprint_library()

    ego=find_ego(world); lead=find_lead(world)

    # 충돌 센서(lead 기준)
    collision_bp = bp_lib.find('sensor.other.collision')
    collided = {"hit": False}
    def on_collision(ev):
        print(f"[COLLISION] lead collided with {ev.other_actor.type_id}")
        collided["hit"] = True
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=lead)
    collision_sensor.listen(on_collision)

    # 초기 시나리오 적용
    sid=int(args.start_scenario); scenario=get_scenario(sid, original_weather)
    reset_positions(world, amap, ego, lead, scenario, sps)
    world.set_weather(original_weather if scenario.weather is None else scenario.weather)

    # === 경로: 동일-차선 경로로 생성 ===
    start_wp = amap.get_waypoint(lead.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
    path = build_same_lane_path(amap, start_wp, max_length_m=2000.0, step_m=2.0)

    # === 제어기 ===
    speed_pid=PID(0.60, 0.05, 0.02)
    stanley = StanleyController(k_e=0.8, k_soft=1.2, steer_rate=1.8, alpha=0.35)
    idx_hint=0

    print(f"[SCENARIO] {scenario.name}  (0/1/2/3 전환, q 종료)")

    # 출발보조/스턱
    KICK_DURATION=1.0; KICK_THROTTLE=0.55; kick_until=0.0
    STUCK_SPEED=0.5; STUCK_TIMEOUT=2.0
    EMERG_KICK=1.2; emerg_until=0.0
    stuck_since=None

    # S3 stop zone 상태
    sz = {"visited": set(), "phase":"none", "t0":0.0, "v0":0.0, "hold_until":0.0}

    # 입력
    cmd_q: "queue.Queue[str]" = queue.Queue()
    start_input_thread(cmd_q)

    last_dt=0.05
    try:
        while True:
            snap = world.wait_for_tick()
            dt = snap.timestamp.delta_seconds if snap else last_dt; last_dt=dt
            now = time.time()

            # 입력 처리
            try: cmd=cmd_q.get_nowait()
            except queue.Empty: cmd=None
            if cmd in ("q","Q"): break
            if cmd in ("0","1","2","3"):
                try: collision_sensor.stop()
                except: pass
                try: collision_sensor.destroy()
                except: pass

                speed_pid.reset(); stanley.reset(); idx_hint=0
                collided["hit"]=False
                sid=int(cmd); scenario=get_scenario(sid, original_weather)
                reset_positions(world, amap, ego, lead, scenario, sps)
                world.set_weather(original_weather if scenario.weather is None else scenario.weather)

                # 경로 재생성(동일 차선)
                start_wp = amap.get_waypoint(lead.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
                path = build_same_lane_path(amap, start_wp, max_length_m=2000.0, step_m=2.0)

                collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=lead)
                collision_sensor.listen(on_collision)

                kick_until=0.0; emerg_until=0.0; stuck_since=None
                sz = {"visited": set(), "phase":"none", "t0":0.0, "v0":0.0, "hold_until":0.0}
                print(f"\n[SCENARIO] {scenario.name}")
                continue

            # 충돌 시 즉시 정지 유지
            if collided["hit"]:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                speed_pid.reset()
                continue

            # 상태
            tf = lead.get_transform(); loc=tf.location; yaw=tf.rotation.yaw
            v = lead.get_velocity(); speed=math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z); speed_kmh=3.6*speed

            # 경로 끝 근처면 연장
            if idx_hint>len(path)-80:
                start_wp = amap.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                path = build_same_lane_path(amap, start_wp, max_length_m=2000.0, step_m=2.0)
                idx_hint = 0

            # s좌표
            s = track_frame.s_on_line(loc)

            # 기본 목표속도
            kmh_cmd = 55.0 if scenario.sid==3 else 60.0

            # 최종 정지
            if s >= scenario.stop_s:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                speed_pid.reset(); stanley.reset()
                continue

            # S3: stop zones 처리(-4 m/s² 감속 → 5초 정지 → 출발 킥)
            if scenario.sid == 3:
                if sz["phase"] == "none":
                    for ss in scenario.stop_zones:
                        if ss in sz["visited"]: continue
                        if abs(s - ss) <= 5.0:
                            print(f"[EVENT] SZ {ss:.0f} m → decel -4 m/s² to stop, hold 5s")
                            sz["visited"].add(ss)
                            sz["phase"]="decel"; sz["t0"]=now; sz["v0"]=speed
                            break
                if sz["phase"] == "decel":
                    t = max(0.0, now - sz["t0"])
                    v_target = max(0.0, sz["v0"] - 4.0*t)  # m/s
                    kmh_cmd = min(kmh_cmd, v_target*3.6)
                    if (v_target <= 0.2) and (speed <= 0.3):
                        sz["phase"]="hold"; sz["hold_until"]=now+5.0
                        speed_pid.reset(); kmh_cmd=0.0
                elif sz["phase"] == "hold":
                    kmh_cmd = 0.0
                    if now >= sz["hold_until"]:
                        sz["phase"]="none"
                        speed_pid.reset()
                        kick_until = now + KICK_DURATION  # 재출발 킥

            # 선형 감속(시나리오별)
            if scenario.decel_start <= s < scenario.stop_s:
                v_allow = MAX_CRUISE_KMH * (scenario.stop_s - s) / max((scenario.stop_s - scenario.decel_start), 1e-3)
                kmh_cmd = min(kmh_cmd, v_allow)

            # 곡률 기반 추가 감속(모든 시나리오 보호용)
            try:
                wp_now = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                nxt = wp_now.next(15.0)
                if nxt:
                    yaw_next = nxt[0].transform.rotation.yaw
                    yaw_diff = abs(deg_wrap180(yaw_next - wp_now.transform.rotation.yaw))
                    if yaw_diff > 30: kmh_cmd = min(kmh_cmd, 28.0)
                    elif yaw_diff > 18: kmh_cmd = min(kmh_cmd, 38.0)
                    elif yaw_diff > 12: kmh_cmd = min(kmh_cmd, 45.0)
            except: pass

            # === 경로 상 최근접 인덱스 & Stanley 조향 ===
            idx_hint = nearest_index_on_path(path, loc, hint=idx_hint, search_window=50)
            steer = StanleyController.step(stanley, path, idx_hint, loc, yaw, speed, dt)

            # 속도 제어 + 보호
            v_ref = kmh_cmd / 3.6
            a_cmd = speed_pid.step(v_ref - speed, dt)
            throttle=0.0; brake=0.0

            # 오버스피드 즉시 제동
            if speed_kmh > kmh_cmd + 5.0:
                throttle = 0.0
                brake = max(brake, 0.35)

            # 출발/스턱 보조 (S3 감속/정지 중에는 킥 비활성)
            allow_kick = (scenario.sid != 3) or (sz["phase"] == "none")
            if allow_kick:
                if (kmh_cmd > 0.1) and (speed < STUCK_SPEED):
                    if stuck_since is None: stuck_since = now
                    elif (now - stuck_since) >= STUCK_TIMEOUT:
                        emerg_until = now + EMERG_KICK
                        stuck_since = None
                else:
                    stuck_since = None

            if now < emerg_until:
                throttle = max(throttle, 0.60); brake=0.0
            elif now < kick_until:
                throttle = max(throttle, KICK_THROTTLE); brake=0.0
            else:
                if kmh_cmd <= 0.1:
                    throttle=0.0; brake=1.0; speed_pid.reset()
                else:
                    if a_cmd >= 0: throttle = clamp(a_cmd, 0.0, 0.65)  # 상한 약간 보수적
                    else: brake = clamp(-a_cmd, 0.0, 1.0)

            lead.apply_control(carla.VehicleControl(throttle=float(throttle),
                                                    steer=float(steer),
                                                    brake=float(brake)))
    finally:
        try:
            collision_sensor.stop(); collision_sensor.destroy()
        except: pass
        try:
            world.set_weather(original_weather)
        except: pass
        print("[CLEANUP] collision sensor 해제 & 날씨 복구 완료")

if __name__ == "__main__":
    main()
