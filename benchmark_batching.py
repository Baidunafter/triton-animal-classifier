"""Сравнение 4 конфигураций Dynamic Batching в Triton."""
import subprocess
import time
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
import load_test  # noqa: E402

CONFIG_PATH = Path("model_repository/image_classifier/config.pbtxt")
MODEL_READY_URL = "http://localhost:8000/v2/models/image_classifier/ready"

# Минимальный конфиг (Triton сам определит input/output из ONNX)
BASE_CONFIG = """name: "image_classifier"
backend: "onnxruntime"
max_batch_size: 32

instance_group [
  {{ count: 1, kind: KIND_AUTO }}
]
{batching}
"""

BATCHING_BLOCKS = {
    "no_batch": "",
    "small": "dynamic_batching { preferred_batch_size: [ 2, 4 ] max_queue_delay_microseconds: 50000 }",
    "medium": "dynamic_batching { preferred_batch_size: [ 4, 8, 16 ] max_queue_delay_microseconds: 100000 }",
    "large": "dynamic_batching { preferred_batch_size: [ 8, 16, 32 ] max_queue_delay_microseconds: 200000 }",
}


def write_config(name):
    CONFIG_PATH.write_text(BASE_CONFIG.format(batching=BATCHING_BLOCKS[name]), encoding="utf-8")


def restart_stack():
    """Рестартим triton + api (api держит gRPC-соединение, его тоже надо обновить)."""
    subprocess.run(["docker", "compose", "restart", "triton", "api"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def wait_ready():
    deadline = time.time() + 120
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            try:
                async with s.get(MODEL_READY_URL, timeout=2) as r:
                    if r.status == 200:
                        await asyncio.sleep(3)  # запас на прогрев API gRPC-клиента
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def main():
    all_results = {}
    for name in ["no_batch", "small", "medium", "large"]:
        print(f"\n=== {name} ===")
        write_config(name)
        restart_stack()
        if not await wait_ready():
            print(f"Таймаут для {name}, пропускаем")
            continue
        all_results[name] = await load_test.run()
        await asyncio.sleep(2)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n=== ИТОГ ===")
    print(f"{'Config':<10}{'RPS':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'fail':>8}")
    for name, r in all_results.items():
        print(f"{name:<10}{r['rps']:>8.1f}{r['latency_p50_ms']:>8.1f}"
              f"{r['latency_p95_ms']:>8.1f}{r['latency_p99_ms']:>8.1f}{r['failed']:>8}")

    # Возвращаем дефолт
    write_config("medium")
    restart_stack()


if __name__ == "__main__":
    asyncio.run(main())
