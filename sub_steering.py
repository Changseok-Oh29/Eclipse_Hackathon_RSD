# sub_steering.py  (Ctrl+C로 종료)
import time
from kuksa_client.grpc import VSSClient

HOST, PORT = "127.0.0.1", 55555
PATH = "Vehicle.ADAS.LK.Steering"

def as_path(e):
    # 다양한 형식 방어: 객체형/.path, .entry, 문자열, dict 등
    if hasattr(e, "path"):
        return e.path
    if hasattr(e, "entry"):
        return e.entry
    if isinstance(e, str):
        return e
    if isinstance(e, dict):
        return e.get("path") or e.get("entry")
    return None

def get_value(client, path):
    v = client.get_current_value(path)
    # v가 dict일 수도, 바로 값일 수도 있음
    if isinstance(v, dict):
        return v.get("value", v)
    return v

def main():
    c = VSSClient(HOST, PORT)
    c.connect()
    print(f"[INFO] Connected to Databroker {HOST}:{PORT}")

    # 일부 버전은 콜백을 받지 않고, '업데이트 묶음' 이터레이터를 반환
    sub = c.subscribe_current_values([PATH])
    print(f"[INFO] Subscribed to {PATH}. Waiting for updates...")

    try:
        for upd in sub:
            # upd가 리스트/이터러블일 수도, 단일 엔트리일 수도 있음
            entries = upd if isinstance(upd, (list, tuple)) else [upd]
            for e in entries:
                p = as_path(e)
                if not p:
                    continue
                val = get_value(c, p)   # ★ 경로만 오면 여기서 현재값을 읽는다
                print(f"[SUB] {p} = {val}")
    except KeyboardInterrupt:
        pass
    finally:
        c.disconnect()

if __name__ == "__main__":
    main()
