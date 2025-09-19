# common/acc_algo.py
# ACC core logic extracted from decision.py and wrapped as a reusable controller.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---- ACC 파라미터 (decision.py에서 가져옴) ----
TARGET_SPEED_KPH             = 80.0
TARGET_SPEED_MPS             = TARGET_SPEED_KPH / 3.6
TIME_HEADWAY                 = 1.8
MIN_GAP                      = 5.0

KP_GAP                       = 0.30
KV_SPEED                     = 0.22
MAX_ACCEL                    = 1.2
MAX_DECEL                    = 5.0
THR_GAIN                     = 0.40
BRK_GAIN                     = 0.35

AEB_ENTER_TTC                = 3.0
AEB_EXIT_TTC                 = 4.5
FOLLOW_ENTER_TTC             = 8.0
FOLLOW_EXIT_TTC              = 10.0
AEB_ACTIVATION_SPEED_THRESHOLD = 2.8
AEB_MIN_HOLD_SEC             = 0.8
MODE_COOLDOWN_SEC            = 0.4

MIN_CONFIRM_FRAMES           = 2
FOLLOW_MAX_DIST_M            = 80.0

# 레이트 리미터(20Hz 기준)
THR_RATE_UP                  = 0.08
THR_RATE_DOWN                = 0.25
BRK_RATE_UP                  = 0.25
BRK_RATE_DOWN                = 0.12

# 접근 바이어스
APPROACH_REL_GAIN            = 0.06
TTC_SLOW_START               = 7.0
TTC_BIAS_GAIN                = 0.05


def acc_longitudinal_control(ego_speed, lead_speed, distance, target_speed_mps):
    desired_gap = MIN_GAP + TIME_HEADWAY * ego_speed
    gap_error = (distance if distance is not None else float('inf')) - desired_gap
    if lead_speed is not None:
        speed_err = (lead_speed - ego_speed)
    else:
        speed_err = (target_speed_mps - ego_speed)
    accel_cmd = KP_GAP * gap_error + KV_SPEED * speed_err
    accel_cmd = max(-MAX_DECEL, min(MAX_ACCEL, float(accel_cmd)))
    if accel_cmd >= 0:
        throttle, brake = min(1.0, THR_GAIN * accel_cmd), 0.0
    else:
        throttle, brake = 0.0, min(1.0, BRK_GAIN * abs(accel_cmd))
    return float(throttle), float(brake)


def rate_limit(prev, desired, max_up, max_down):
    delta = desired - prev
    if delta >  max_up:   desired = prev + max_up
    if delta < -max_down: desired = prev - max_down
    return float(np.clip(desired, 0.0, 1.0))


@dataclass
class ACCInputs:
    distance: float | None
    rel_speed: float      # +면 접근중
    ttc: float | None
    has_target: bool
    lead_speed_est: float | None = None


class ACCController:
    """
    상태기계 + 종방향 제어 + 레이트 리미트까지 한 번에 처리.
    decision.py의 ACC 블록을 그대로 이식/래핑.
    """
    def __init__(self, apply_hz: float = 20.0):
        self.apply_hz = max(1e-3, float(apply_hz))
        self.mode = "CRUISE"
        self.last_mode_change_time = -1e9
        self.safe_stop_locked = False
        self.last_thr = 0.0
        self.last_brk = 0.0

    def step(self, sim_time: float, ego_speed_mps: float, acc: ACCInputs):
        # 입력 전처리
        distance = float(acc.distance) if (acc.distance is not None) else float('inf')
        ttc = float(acc.ttc) if (acc.ttc is not None) else float('inf')
        rel_v = float(acc.rel_speed)

        # FOLLOW 표적 게이팅
        target_ready = (
            bool(acc.has_target)
            and (distance < FOLLOW_MAX_DIST_M)
            and np.isfinite(ttc)
            and (ttc < FOLLOW_ENTER_TTC)
        )

        tsc = sim_time - self.last_mode_change_time
        throttle_des = 0.0
        brake_des = 0.0

        # ---- 상태 전이 ----
        if self.mode == "CRUISE":
            if target_ready and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time = "FOLLOW", sim_time
            if target_ready and (ttc < AEB_ENTER_TTC) and (ego_speed_mps > AEB_ACTIVATION_SPEED_THRESHOLD) and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time, self.safe_stop_locked = "AEB", sim_time, False

        elif self.mode == "FOLLOW":
            if target_ready and (ttc < AEB_ENTER_TTC) and (ego_speed_mps > AEB_ACTIVATION_SPEED_THRESHOLD) and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time, self.safe_stop_locked = "AEB", sim_time, False
            elif ((not target_ready) or (ttc >= FOLLOW_EXIT_TTC) or (distance == float('inf'))) and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time = "CRUISE", sim_time

        elif self.mode == "AEB":
            if (not self.safe_stop_locked) and (distance <= 6.0):
                self.safe_stop_locked = True
            if self.safe_stop_locked:
                throttle_des, brake_des = 0.0, 1.0
            else:
                throttle_des, brake_des = 0.0, min(1.0, max(0.0, distance / max(1e-3, 6.0)))
            if (sim_time - self.last_mode_change_time) >= AEB_MIN_HOLD_SEC and (ttc > AEB_EXIT_TTC):
                self.mode = "FOLLOW" if target_ready else "CRUISE"
                self.last_mode_change_time = sim_time
                self.safe_stop_locked = False

        # ---- FOLLOW/CRUISE 물리 제어 ----
        if self.mode != "AEB":
            lead_speed_for_acc = None
            if target_ready:
                # rel_v(+접근) → lead 절대속도 = ego - rel_v
                lead_speed_for_acc = max(0.0, ego_speed_mps - rel_v)

            throttle_des, brake_des = acc_longitudinal_control(
                ego_speed_mps, lead_speed_for_acc, distance, TARGET_SPEED_MPS
            )

            # 접근 바이어스(FOLLOW에서만)
            if self.mode == "FOLLOW":
                if rel_v > 0.0:
                    brake_des    = min(1.0, brake_des + APPROACH_REL_GAIN * rel_v)
                    throttle_des = max(0.0, throttle_des - APPROACH_REL_GAIN * rel_v)
                if np.isfinite(ttc) and ttc < TTC_SLOW_START:
                    bias = TTC_BIAS_GAIN * (TTC_SLOW_START - ttc)
                    throttle_des = max(0.0, throttle_des - bias)
                    brake_des    = min(1.0, brake_des + bias)

            # 크루즈 부스트
            if (not target_ready):
                speed_gap = max(0.0, TARGET_SPEED_MPS - ego_speed_mps)
                boost = np.clip(0.12 + 0.06 * speed_gap, 0.12, 0.60)
                throttle_des = max(throttle_des, float(boost))
                brake_des = 0.0

        # ---- 레이트 리미트 & 결과 ----
        thr_cmd = rate_limit(self.last_thr, throttle_des, THR_RATE_UP, THR_RATE_DOWN)
        brk_cmd = rate_limit(self.last_brk, brake_des, BRK_RATE_UP, BRK_RATE_DOWN)
        self.last_thr, self.last_brk = thr_cmd, brk_cmd

        debug = {
            "target_ready": bool(target_ready),
            "distance": distance if np.isfinite(distance) else float('inf'),
            "ttc": ttc if np.isfinite(ttc) else float('inf'),
            "rel_v": rel_v,
        }
        return float(thr_cmd), float(brk_cmd), self.mode, debug
