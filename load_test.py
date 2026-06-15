"""Нагрузочный тест FastAPI gateway: подаёт картинки из animals/ циклически."""
import asyncio
import time
import statistics
import json
import random
import sys
from pathlib import Path

import aiohttp

API_URL = "http://localhost:8080/predict"
IMAGES_DIR = "animals"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOTAL_REQUESTS = 1000
CONCURRENCY = 10
TIMEOUT_S = 60


def load_images():
    p = Path(IMAGES_DIR)
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"ОШИБКА: в {IMAGES_DIR}/ нет картинок."); sys.exit(1)
    random.seed(42)
    random.shuffle(files)
    return [(f.name, f.read_bytes()) for f in files]


async def worker(session, images, queue, results):
    while True:
        idx = await queue.get()
        if idx is None:
            queue.task_done(); return
        name, image_bytes = images[idx % len(images)]
        form = aiohttp.FormData()
        form.add_field("file", image_bytes, filename=name, content_type="image/jpeg")
        t0 = time.perf_counter()
        try:
            async with session.post(API_URL, data=form, timeout=TIMEOUT_S) as resp:
                await resp.read()
                ok = resp.status == 200
        except Exception:
            ok = False
        results.append((ok, time.perf_counter() - t0))
        queue.task_done()


async def run():
    images = load_images()
    print(f"Загружено {len(images)} картинок, запросов={TOTAL_REQUESTS}, concurrency={CONCURRENCY}")

    queue = asyncio.Queue()
    for i in range(TOTAL_REQUESTS): queue.put_nowait(i)
    for _ in range(CONCURRENCY): queue.put_nowait(None)

    results = []
    conn = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        t_start = time.perf_counter()
        workers = [asyncio.create_task(worker(session, images, queue, results))
                   for _ in range(CONCURRENCY)]
        await queue.join()
        for w in workers: await w
        total_time = time.perf_counter() - t_start

    ok_count = sum(1 for ok, _ in results if ok)
    latencies = sorted([dt for ok, dt in results if ok])

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    stats = {
        "total_requests": TOTAL_REQUESTS,
        "concurrency": CONCURRENCY,
        "successful": ok_count,
        "failed": len(results) - ok_count,
        "total_time_s": round(total_time, 2),
        "rps": round(ok_count / total_time, 1),
        "latency_avg_ms": round(statistics.mean(latencies) * 1000, 1),
        "latency_p50_ms": round(pct(0.50) * 1000, 1),
        "latency_p95_ms": round(pct(0.95) * 1000, 1),
        "latency_p99_ms": round(pct(0.99) * 1000, 1),
    }
    print("\n=== Результаты ===")
    for k, v in stats.items(): print(f"  {k:20s} {v}")
    with open("load_results.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    asyncio.run(run())
