# -*- coding: utf-8 -*-
# carla_cam_zenoh_pub.py (확인용 로깅 강화)
import sys, os, time
import numpy as np
import cv2
import zenoh

# --- CARLA egg 경로 ---
EGG = os.path.expanduser('~/Carla_0.9.14/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg')
if not os.path.exists(EGG):
    raise FileNotFoundError(EGG)
sys.path.append(EGG)
import carla

CARLA_HOST, CARLA_PORT = os.environ.get("CARLA_HOST","127.0.0.1"), int(os.environ.get("CARLA_PORT","2000"))
Z_KEY = os.environ.get("Z_KEY","carla/cam/front")
WIDTH, HEIGHT, FOV, JPEG_QUALITY = int(os.environ.get("IMG_W","640")), int(os.environ.get("IMG_H","480")), 90, int(os.environ.get("JPEG_Q","80"))
SENSOR_TICK = float(os.environ.get("SENSOR_TICK","0.05"))  # 20Hz
SAVE_EVERY = int(os.environ.get("SAVE_EVERY","10"))        # N프레임마다 /tmp 저장

def find_hero(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.attributes.get('role_name','') == 'hero':
            return a
    return None

def main():
    # zenoh 세션 (라우터 연결 강제하고 싶으면: export ZENOH_CONNECT=tcp/127.0.0.1:7447)
    conf = zenoh.Config()
    session = zenoh.open(conf)          # ← 환경변수(ZENOH_CONNECT/ LISTEN) 자동 인식
    pub = session.declare_publisher(Z_KEY)

    # CARLA
    client = carla.Client(CARLA_HOST, CARLA_PORT); client.set_timeout(5.0)
    world = client.get_world()

    hero = None
    for _ in range(60):
        hero = find_hero(world)
        if hero: break
        print("[CARLA] waiting hero...")
        time.sleep(1.0)
    if not hero: raise RuntimeError("hero not found")
    print("[CARLA] hero id:", hero.id)

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(WIDTH))
    bp.set_attribute('image_size_y', str(HEIGHT))
    bp.set_attribute('fov', str(FOV))
    bp.set_attribute('sensor_tick', str(SENSOR_TICK))  # 중요

    cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))
    cam = world.spawn_actor(bp, cam_tf, attach_to=hero)

    frame_cnt = 0
    def on_image(image):
        nonlocal frame_cnt
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
            ok, buf = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                print("[PUB] imencode failed")
                return
            pub.put(buf.tobytes())
            frame_cnt += 1
            if frame_cnt % SAVE_EVERY == 0:
                cv2.imwrite("/tmp/pub_latest.jpg", arr)
            print(f"[PUB] sent bytes={len(buf)} frame={image.frame} sim_ts={image.timestamp:.3f}")
        except Exception as e:
            print("[PUB] callback error:", e)

    cam.listen(on_image)
    print("[CARLA] camera streaming (attached to hero). Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop(); cam.destroy(); session.close()

if __name__ == "__main__":
    main()
