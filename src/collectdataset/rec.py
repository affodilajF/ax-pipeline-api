import sys
import os
import time
import asyncio
import cv2
import numpy as np

# Tambahkan path ke folder yang ada m3axpidataset.so
sys.path.append("/root/librarychange/axpipeline/ax-pipeline-api")

try:
    import m3axpidataset
except ImportError as e:
    print(f"❌ Gagal import m3axpidataset: {e}")
    exit(1)

# Folder hasil simpan
SAVE_DIR = "/root/coba4/4"
os.makedirs(SAVE_DIR, exist_ok=True)

async def stream_camera():
    # Inisialisasi kamera
    try:
        m3axpidataset.camera(SysCase=2)
        m3axpidataset.load("/home/config/yolov8.json")
        time.sleep(1)
        print("✅ Kamera siap")
    except Exception as e:
        print(f"❌ Gagal inisialisasi kamera: {e}")
        return

    frame_count = 0
    last_save_time = time.time()

    while True:
        try:
            frame_npu = m3axpidataset.capture()

            if frame_npu is None:
                await asyncio.sleep(0.01)
                continue
            
            h, w, c = frame_npu[0], frame_npu[1], frame_npu[2]
            buf = frame_npu[3]  # ini bytes object dari py::bytes

            raw = np.frombuffer(buf, dtype=np.uint8)
            raw = raw.reshape((h, w, c))

            raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            # raw_bgr = raw

            now = time.time()
            if now - last_save_time >= 1.0:
                filename = os.path.join(SAVE_DIR, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(filename, raw_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"💾 Saved {filename}")
                frame_count += 1
                last_save_time = now

            await asyncio.sleep(0.001)  # kecilkan CPU usage

        except Exception as e:
            print(f"❌ Error capture: {e}, retry in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(stream_camera())
    except KeyboardInterrupt:
        print("🛑 Streaming dihentikan")
    finally:
        cv2.destroyAllWindows()
