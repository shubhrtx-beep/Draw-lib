"""
Draw-lib Performance Benchmark Suite
=====================================
Measures key performance metrics for Draw-lib without requiring a display.
All tests run headlessly or with minimal Qt setup.
"""
import sys
import time
import gc
import os
import json
import statistics

# Ensure parent dir is on path
_DRAW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DRAW_DIR not in sys.path:
    sys.path.insert(0, _DRAW_DIR)


def measure_import_time(n_trials=5):
    """Measure cold import time by spawning subprocesses."""
    import subprocess
    times = []
    for _ in range(n_trials):
        cmd = [sys.executable, "-c",
               "import time; t=time.perf_counter(); import Draw; print(time.perf_counter()-t)"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_DRAW_DIR)
        if result.returncode == 0:
            try:
                t = float(result.stdout.strip().split('\n')[-1])
                times.append(t)
            except (ValueError, IndexError):
                pass
    if not times:
        return {"mean_s": -1, "median_s": -1, "min_s": -1, "max_s": -1, "trials": 0}
    return {
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "trials": len(times),
    }


def measure_shape_creation(counts=(100, 1000, 5000)):
    """Measure ShapeDef creation time."""
    from Draw._shapes import ShapeDef
    from PySide6.QtGui import QColor

    results = {}
    for count in counts:
        gc.disable()
        start = time.perf_counter()
        shapes = []
        for i in range(count):
            s = ShapeDef(
                vertices=4,
                size_raw=[50, 50],
                border_radius_raw=0,
                x=i % 100 * 55,
                y=i // 100 * 55,
                align=None,
                rotation=0.0,
                color=QColor(0, 200, 200),
                border_color=QColor(255, 255, 255),
                border_width=0,
                border_style="solid",
                opacity=100,
                curve_mode="line",
                bend=[],
                bend_amount=0.0,
                warp=None,
                exclude=[],
                symmetry=None,
                hitbox_mode=None,
                hit_box="shape",
                custom=None,
                z=0,
                overlap=True,
                flow=None,
                ip=f"bench_{i}",
            )
            shapes.append(s)
        elapsed = time.perf_counter() - start
        gc.enable()
        results[f"{count}_shapes"] = {
            "total_s": elapsed,
            "per_shape_us": (elapsed / count) * 1e6,
        }
    return results


def measure_z_sort(counts=(100, 1000, 5000, 10000)):
    """Measure z-sort overhead."""
    import random

    class FakeShape:
        __slots__ = ('z', 'ip')
        def __init__(self, z, ip):
            self.z = z
            self.ip = ip

    results = {}
    for count in counts:
        shapes = [FakeShape(z=random.randint(0, 100), ip=f"s{i}") for i in range(count)]
        sorted(shapes, key=lambda s: -s.z)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = sorted(shapes, key=lambda s: -s.z)
            times.append(time.perf_counter() - start)
        
        results[f"{count}_shapes"] = {
            "mean_us": statistics.mean(times) * 1e6,
            "median_us": statistics.median(times) * 1e6,
        }
    return results


def measure_color_parse(n_iterations=10000):
    """Measure color parsing overhead."""
    from Draw._super import _parse_color_to_rgba

    colors = ["cyan", "#FF5500", "rgb(100,200,50)", "white", "black", "#112233"]
    
    gc.disable()
    start = time.perf_counter()
    for _ in range(n_iterations):
        for c in colors:
            _parse_color_to_rgba(c, 100)
    elapsed = time.perf_counter() - start
    gc.enable()
    
    total_calls = n_iterations * len(colors)
    return {
        "total_calls": total_calls,
        "total_s": elapsed,
        "per_call_us": (elapsed / total_calls) * 1e6,
    }


def measure_expression_parse(n_iterations=5000):
    """Measure expression parsing/evaluation overhead."""
    from Draw._calculator import eval_expression

    expressions = [
        "sin(time * 3) * 50 + 50",
        "cos(time * 2) * 100",
        "time * 60",
        "lerp(0, 255, time / 10)",
        "abs(sin(time)) * 100",
    ]

    gc.disable()
    start = time.perf_counter()
    for _ in range(n_iterations):
        for expr in expressions:
            try:
                eval_expression(expr, {"time": 1.5, "x": 100, "y": 200})
            except Exception:
                pass
    elapsed = time.perf_counter() - start
    gc.enable()
    
    total_calls = n_iterations * len(expressions)
    return {
        "total_calls": total_calls,
        "total_s": elapsed,
        "per_call_us": (elapsed / total_calls) * 1e6,
    }


def measure_spatial_grid(counts=(100, 1000, 5000, 10000)):
    """Measure spatial grid insert/update/query performance."""
    from Draw._optimize import SpatialGridIndex

    results = {}
    for count in counts:
        grid = SpatialGridIndex(cell_size=100.0)
        
        gc.disable()
        start = time.perf_counter()
        for i in range(count):
            x = (i % 100) * 55.0
            y = (i // 100) * 55.0
            grid.insert(i, (x, y, 50.0, 50.0))
        insert_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for i in range(count):
            x = (i % 100) * 55.0
            y = (i // 100) * 55.0
            grid.update(i, (x, y, 50.0, 50.0))
        update_nochange_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for i in range(count):
            x = (i % 100) * 55.0 + 1.0
            y = (i // 100) * 55.0 + 1.0
            grid.update(i, (x, y, 50.0, 50.0))
        update_change_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(1000):
            grid.query_point(250.0, 250.0)
        query_time = time.perf_counter() - start
        gc.enable()
        
        results[f"{count}_items"] = {
            "insert_total_us": insert_time * 1e6,
            "insert_per_item_us": (insert_time / count) * 1e6,
            "update_nochange_total_us": update_nochange_time * 1e6,
            "update_nochange_per_item_us": (update_nochange_time / count) * 1e6,
            "update_change_total_us": update_change_time * 1e6,
            "update_change_per_item_us": (update_change_time / count) * 1e6,
            "query_point_1k_us": query_time * 1e6,
        }
    return results


def measure_polygon_generation(counts=(100, 1000, 5000)):
    """Measure polygon vertex generation overhead."""
    from Draw._shapes import _regular_polygon_points

    results = {}
    for count in counts:
        gc.disable()
        start = time.perf_counter()
        for i in range(count):
            cx, cy = 100.0 + i * 0.01, 100.0
            _regular_polygon_points(cx, cy, 50.0, 50.0, 4)
        elapsed = time.perf_counter() - start
        gc.enable()
        results[f"{count}_quads"] = {
            "total_us": elapsed * 1e6,
            "per_shape_us": (elapsed / count) * 1e6,
        }
    return results


def measure_font_creation(n_iterations=5000):
    """Measure QFont + QFontMetricsF creation overhead."""
    from PySide6.QtGui import QFont, QFontMetricsF

    gc.disable()
    start = time.perf_counter()
    for _ in range(n_iterations):
        font = QFont("Segoe UI")
        font.setPixelSize(16)
        font.setBold(False)
        font.setItalic(False)
        fm = QFontMetricsF(font)
    elapsed = time.perf_counter() - start
    gc.enable()

    return {
        "total_calls": n_iterations,
        "total_s": elapsed,
        "per_call_us": (elapsed / n_iterations) * 1e6,
    }


def run_all():
    """Run all benchmarks and report results."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    print("=" * 70)
    print("Draw-lib Performance Benchmark Suite")
    print("=" * 70)

    results = {}

    print("\n[1/7] Measuring import time...")
    results["import_time"] = measure_import_time()
    print(f"  Import time: {results['import_time']['median_s']*1000:.1f} ms (median)")

    print("\n[2/7] Measuring shape creation...")
    results["shape_creation"] = measure_shape_creation()
    for k, v in results["shape_creation"].items():
        print(f"  {k}: {v['total_s']*1000:.1f} ms total, {v['per_shape_us']:.1f} us/shape")

    print("\n[3/7] Measuring z-sort...")
    results["z_sort"] = measure_z_sort()
    for k, v in results["z_sort"].items():
        print(f"  {k}: {v['mean_us']:.1f} us mean")

    print("\n[4/7] Measuring color parse...")
    results["color_parse"] = measure_color_parse()
    print(f"  {results['color_parse']['per_call_us']:.2f} us/call ({results['color_parse']['total_calls']} calls)")

    print("\n[5/7] Measuring expression parse...")
    results["expression_parse"] = measure_expression_parse()
    print(f"  {results['expression_parse']['per_call_us']:.2f} us/call ({results['expression_parse']['total_calls']} calls)")

    print("\n[6/7] Measuring spatial grid...")
    results["spatial_grid"] = measure_spatial_grid()
    for k, v in results["spatial_grid"].items():
        print(f"  {k}:")
        print(f"    insert: {v['insert_per_item_us']:.2f} us/item")
        print(f"    update (no change): {v['update_nochange_per_item_us']:.2f} us/item")
        print(f"    update (changed): {v['update_change_per_item_us']:.2f} us/item")

    print("\n[7/7] Measuring polygon generation & font creation...")
    results["polygon_generation"] = measure_polygon_generation()
    for k, v in results["polygon_generation"].items():
        print(f"  {k}: {v['per_shape_us']:.2f} us/shape")

    results["font_creation"] = measure_font_creation()
    print(f"  Font creation: {results['font_creation']['per_call_us']:.2f} us/call")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    
    print("\nEstimated per-frame cost for 1,000 shapes (at 60 FPS = 16.67ms budget):")
    z_1k = results["z_sort"].get("1000_shapes", {}).get("mean_us", 0)
    color_per = results["color_parse"]["per_call_us"]
    poly_per = results["polygon_generation"].get("1000_quads", {}).get("per_shape_us", 0)
    grid_per = results["spatial_grid"].get("1000_items", {}).get("update_change_per_item_us", 0)
    font_per = results["font_creation"]["per_call_us"]
    
    z_cost = z_1k
    color_cost = color_per * 1000
    poly_cost = poly_per * 1000
    grid_cost = grid_per * 1000
    total_overhead_us = z_cost + color_cost + poly_cost + grid_cost
    
    print(f"  Z-sort:          {z_cost:>10.0f} us")
    print(f"  Color parse:     {color_cost:>10.0f} us")
    print(f"  Polygon gen:     {poly_cost:>10.0f} us")
    print(f"  Spatial grid:    {grid_cost:>10.0f} us")
    print(f"  --------------------------------")
    print(f"  Total overhead:  {total_overhead_us:>10.0f} us = {total_overhead_us/1000:.1f} ms")
    print(f"  Frame budget:    {16670:>10} us = 16.7 ms")
    print(f"  Overhead ratio:  {total_overhead_us/16670*100:.1f}%")
    
    return results


if __name__ == "__main__":
    run_all()
