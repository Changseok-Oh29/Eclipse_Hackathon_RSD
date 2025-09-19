# zcopy_pub.py — RAW(BGRA) zero-copy 경로, 기능 동일 / 로그·메타·튜닝변수 보강
import time, json, zenoh, os

# --- CARLA import (same as now) ---
try:
    import carla
except ImportError:
    import glob, sys
    eggs = sorted(glob.glob(os.path.expanduser('~/carla-0.9.15/PythonAPI/carla/dist/carla-*py3*.egg')))
    if eggs: sys.path.append(eggs[-1])
    import carla

# ===== 튜닝 파라미터 =====
IMG_W        = int(os.environ.get("IMG_W", "640"))   # ← 필요시 960/800/640 등으로 낮추기
IMG_H        = int(os.environ.get("IMG_H", "480"))
SENSOR_TICK  = float(os.environ.get("SENSOR_TICK", "0.05"))  # 20Hz (=0.05s). 15fps면 0.0667

# ===== zenoh =====
cfg = zenoh.Config()

#sess = zenoh.open(cfg) 최후에 주석 해제
#pub  = sess.declare_publisher('carla/cam/front')
try:
    cfg.insert_json5('mode', '"client"')
    cfg.insert_json5('connect/endpoints', '["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json('mode', '"client"')
    cfg.insert_json('connect/endpoints', '["tcp/127.0.0.1:7447"]')

sess = zenoh.open(cfg)
pub  = sess.declare_publisher('carla/cam/front')

# ===== CARLA =====
cli = carla.Client('127.0.0.1', 2000); cli.set_timeout(5.0)
world = cli.get_world()
bp = world.get_blueprint_library().find('sensor.camera.rgb')
bp.set_attribute('image_size_x', str(IMG_W))
bp.set_attribute('image_size_y', str(IMG_H))
bp.set_attribute('fov', '90')
bp.set_attribute('sensor_tick', str(SENSOR_TICK))  # ★ 템포 고정
cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=1.6)))

def on_img(img: carla.Image):
    buf = memoryview(img.raw_data)  # BGRA 원버퍼
    att = json.dumps({
        'w': img.width, 'h': img.height, 'c': 4,
        'format': 'bgra8', 'stride': img.width * 4,
        'frame': int(img.frame), 'sim_ts': float(img.timestamp),
        'pub_ts': time.time()
    }).encode('utf-8')
    pub.put(bytes(buf), attachment=att)  # 파이썬 특성상 1회 복사 허용
    print(f"[PUB] RAW bytes={len(buf)} frame={img.frame} sim_ts={img.timestamp:.3f}")

cam.listen(on_img)
try:
    while True: time.sleep(1)
finally:
    cam.stop(); cam.destroy(); sess.close()
