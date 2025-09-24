#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lead_scenario.py — Pure Pursuit + 시나리오 속도제어

- S0: spawn_idx=20 혼합 루트, 60 km/h, 커브 감속, 650→700 정지
- S1: 직선, 기본 날씨, 60 km/h, 550→600 정지
- S2: 직선, Heavy Rain, 60 km/h, 550→600 정지
- S3: 직선, WetNoon, 55 km/h, stop zones(200/400m, -4m/s² 감속 후 5s 정지 → 재출발), 최종 550→600 정지
- 리드 차량 충돌 시 즉시 풀브레이크 정지 유지
- 리드 차량 타이어 마찰계수 μ는 항상 dry 고정
- TrafficManager/autopilot 사용 안 함
- 키보드 입력: 0/1/2/3 시나리오 전환, q 종료
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
# 시나리오
# =========================
@dataclass
class Scenario:
    sid: int
    name: str
    target_kmh: float
    route_type: str  # "mixed" | "straight"
    weather: Optional[carla.WeatherParameters]
    decel_start: float
    stop_s: float
    stop_zones: List[float] = field(default_factory=list)

def get_scenario(sid: int, original_weather: carla.WeatherParameters) -> Scenario:
    if sid == 0:
        return Scenario(0,"S0",60.0,"mixed",None,650.0,700.0)
    if sid == 1:
        return Scenario(1,"S1",60.0,"straight",None,550.0,600.0)
    if sid == 2:
        heavy = carla.WeatherParameters(cloudiness=85.0,precipitation=90.0,
                    precipitation_deposits=80.0,wetness=90.0,wind_intensity=0.4,
                    sun_azimuth_angle=20.0,sun_altitude_angle=55.0,
                    fog_density=8.0)
        return Scenario(2,"S2",60.0,"straight",heavy,550.0,600.0)
    if sid == 3:
        return Scenario(3,"S3",55.0,"straight",getattr(carla.WeatherParameters,"WetNoon"),
                        550.0,600.0,[200.0,400.0])
    return get_scenario(0, original_weather)

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
# 스폰
# =========================
def spawn_vehicle_at(world: carla.World, bp: carla.ActorBlueprint, tf: carla.Transform) -> carla.Vehicle:
    for dz in (0.0,0.5,1.0):
        tf_try=carla.Transform(carla.Location(tf.location.x,tf.location.y,tf.location.z+dz),tf.rotation)
        v=world.try_spawn_actor(bp, tf_try)
        if v: return v
    raise RuntimeError("spawn 실패")

def compute_straight_spawn(amap: carla.Map, sps, base_index: int, back_m: float) -> carla.Transform:
    tf0 = sps[min(max(0,base_index),len(sps)-1)]
    wp = amap.get_waypoint(tf0.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    prevs = wp.previous(back_m)
    base_wp = prevs[0] if prevs else wp
    return base_wp.transform

# =========================
# 메인
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--port",type=int,default=2000)
    ap.add_argument("--spawn_idx",type=int,default=20)
    args=ap.parse_args()

    client=carla.Client(args.host,args.port); client.set_timeout(10.0)
    world=client.get_world(); amap=world.get_map()
    bp_lib=world.get_blueprint_library(); sps=amap.get_spawn_points()
    original_weather=world.get_weather()

    # 충돌 플래그
    collided_flag={"hit":False}

    def spawn_lead(scn:Scenario):
        lead_bp=(bp_lib.filter('vehicle.audi.tt') or bp_lib.filter('vehicle.*'))[0]
        lead_bp.set_attribute('role_name','lead')
        if scn.route_type=="mixed":
            tf = sps[min(max(0,args.spawn_idx),len(sps)-1)]
            wp=amap.get_waypoint(tf.location,project_to_road=True,lane_type=carla.LaneType.Driving)
            return spawn_vehicle_at(world,lead_bp,wp.transform)
        else:
            tf_st=compute_straight_spawn(amap,sps,53,200.0)
            return spawn_vehicle_at(world,lead_bp,tf_st)

    def apply_tire_friction(vehicle,tire_mu:float):
        pc=vehicle.get_physics_control()
        for w in pc.wheels: w.tire_friction=tire_mu
        vehicle.apply_physics_control(pc)

    # 초기 시나리오
    scenario=get_scenario(0,original_weather)
    lead=spawn_lead(scenario)
    apply_tire_friction(lead,3.0)

    # 충돌 센서
    col_bp=bp_lib.find('sensor.other.collision')
    col_sensor=world.spawn_actor(col_bp,carla.Transform(),attach_to=lead)
    def on_collision(event):
        print(f"[COLLISION] 리드 차량 충돌 발생 with {event.other_actor.type_id}")
        collided_flag["hit"]=True
    col_sensor.listen(on_collision)

    world.set_weather(original_weather if scenario.weather is None else scenario.weather)
    path=build_forward_path(amap,lead.get_transform())
    speed_pid=PID(0.6,0.05,0.02)
    idx_hint=0
    lookahead_min,lookahead_max=12.0,25.0

    cmd_q: "queue.Queue[str]"=queue.Queue()
    def _inp():
        while True:
            s=sys.stdin.readline().strip()
            if s in ("0","1","2","3","q","Q"): cmd_q.put(s)
    threading.Thread(target=_inp,daemon=True).start()

    last_dt=0.05
    stop_state={"phase":"none","visited":set(),"hold_until":0}
    kick_until=0.0

    try:
        while True:
            snap=world.wait_for_tick()
            dt=snap.timestamp.delta_seconds if snap else last_dt; last_dt=dt
            now=time.time()

            try: cmd=cmd_q.get_nowait()
            except queue.Empty: cmd=None
            if cmd in ("q","Q"): break

            if cmd in ("0","1","2","3"):
                for a in world.get_actors().filter("vehicle.*"): 
                    try:a.destroy()
                    except: pass
                world.set_weather(original_weather)
                scenario=get_scenario(int(cmd),original_weather)
                lead=spawn_lead(scenario)
                apply_tire_friction(lead,3.0)
                path=build_forward_path(amap,lead.get_transform())
                speed_pid.reset(); idx_hint=0
                stop_state={"phase":"none","visited":set(),"hold_until":0}
                collided_flag["hit"]=False
                col_sensor=world.spawn_actor(col_bp,carla.Transform(),attach_to=lead)
                col_sensor.listen(on_collision)
                world.set_weather(original_weather if scenario.weather is None else scenario.weather)
                print(f"[SCENARIO] {scenario.name}")
                continue

            if collided_flag["hit"]:
                lead.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=1))
                speed_pid.reset(); continue

            tf=lead.get_transform(); loc=tf.location; yaw=tf.rotation.yaw
            v=lead.get_velocity(); speed=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z); speed_kmh=3.6*speed

            if idx_hint>len(path)-50:
                path=build_forward_path(amap,tf); idx_hint=0

            # stop zone 처리
            if stop_state["phase"]=="hold":
                if now>=stop_state["hold_until"]:
                    stop_state["phase"]="none"; speed_pid.reset(); kick_until=now+1.0
                else:
                    lead.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=1)); continue
            elif stop_state["phase"].startswith("decel"):
                t=now-stop_state["t0"]
                v_target=max(0.0,stop_state["v0"]-4.0*t)
                kmh_cmd=v_target*3.6
                if v_target<=0.2 and speed<=0.3:
                    stop_state["phase"]="hold"; stop_state["hold_until"]=now+5.0
                    speed_pid.reset(); kmh_cmd=0.0
                else:
                    pass
            else:
                kmh_cmd=scenario.target_kmh
                for sz in scenario.stop_zones:
                    if sz in stop_state["visited"]: continue
                    # 직선 경로 s 좌표 대신 단순 이동거리 사용
                    dist=lead.get_transform().location.distance(path[0].transform.location)
                    if abs(dist-sz)<=5.0:
                        print(f"[EVENT] Stop zone at {sz:.1f} m → decel -4m/s²")
                        stop_state["visited"].add(sz)
                        stop_state={"phase":"decel","t0":now,"v0":speed,"visited":stop_state["visited"]}
                        kmh_cmd=scenario.target_kmh

            # 최종 정지
            dist=lead.get_transform().location.distance(path[0].transform.location)
            if dist>=scenario.stop_s:
                lead.apply_control(carla.VehicleControl(throttle=0,steer=0,brake=1))
                speed_pid.reset(); continue

            # 감속 구간
            if scenario.decel_start<=dist<scenario.stop_s:
                v_allow=60.0*(scenario.stop_s-dist)/max((scenario.stop_s-scenario.decel_start),1e-3)
                kmh_cmd=min(kmh_cmd,v_allow)

            # Pure Pursuit
            lookahead=clamp(lookahead_min+(lookahead_max-lookahead_min)*(speed_kmh/80.0),lookahead_min,lookahead_max)
            idx_hint,tf_tgt=find_lookahead_target(path,loc,lookahead,idx_hint)
            tgt=tf_tgt.location
            dx,dy=tgt.x-loc.x,tgt.y-loc.y
            yaw_rad=math.radians(yaw); fx,fy=math.cos(yaw_rad),math.sin(yaw_rad)
            y_left=-dx*fy+dy*fx; Ld=max(3.0,lookahead)
            steer=clamp(1.2*(2.0*y_left)/(Ld*Ld),-1.0,1.0)

            v_ref=kmh_cmd/3.6
            a_cmd=speed_pid.step(v_ref-speed,dt)
            throttle=0; brake=0
            if now<kick_until: throttle=0.55; brake=0
            else:
                if kmh_cmd<=0.1: throttle=0; brake=1; speed_pid.reset()
                else:
                    if a_cmd>=0: throttle=clamp(a_cmd,0,0.7)
                    else: brake=clamp(-a_cmd,0,1)
            lead.apply_control(carla.VehicleControl(throttle=float(throttle),steer=float(steer),brake=float(brake)))
    finally:
        world.set_weather(original_weather)
        for a in world.get_actors().filter("vehicle.*"):
            try:a.destroy()
            except: pass
        print("[CLEANUP] 종료 완료")

if __name__=="__main__":
    main()
