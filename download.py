#!/usr/bin/env python3
"""Download 20 representative MovingDrone sequences, skipping depth by default.

Uses only Python standard library; respects HTTP_PROXY / HTTPS_PROXY.
Run: python download_movingdrone20.py --root /mnt/Data/RIght/MovingDrone_20

Sources:
  https://github.com/cvg/orthotrack/blob/main/dataset/MovingDrone.py
  https://webshare.cvg.cit.tum.de/g/papers/Dhaouadi/MovingDrone/splits.json
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/MovingDrone"
# 16 different named scene families + 4 distinct held-out locations, 20 total.
# split names were confirmed against the official MovingDrone splits.json.
SCENES = [
    ("airport", "airport10", "train"),
    ("berliner_dom", "berliner_dom4", "test_inPlace"),
    ("bundestag", "bundestag1", "train"),
    ("fernsehturm", "fernsehturm1", "val"),
    ("industrial", "industrial9", "test_inPlace"),
    ("mitte", "mitte21", "test_inPlace"),
    ("olympiastadion", "olympiastadion1", "train"),
    ("pier", "pier1", "train"),
    ("potsdamerplatz", "potsdamerplatz11", "val"),
    ("road", "road3", "val"),
    ("rural", "rural1", "train"),
    ("schloss_charlottenburg", "schloss_charlottenburg10", "test_inPlace"),
    ("siegessaeule", "siegessaeule1", "train"),
    ("suburban", "suburban9", "test_inPlace"),
    ("tu_berlin", "tu_berlin1", "train"),
    ("zoologischer_garten", "zoologischer_garten1", "train"),
    ("outplace_bridge", "Lohmuehlenbruecke", "test_outPlace"),
    ("outplace_street", "elsenstr", "test_outPlace"),
    ("outplace_estrel", "estrel", "test_outPlace"),
    ("outplace_wiener", "wienerstr", "test_outPlace"),
]
assert len(SCENES) == 20 and len({x[1] for x in SCENES}) == 20

OPENER = urllib.request.build_opener()
PRINT_LOCK = threading.Lock()

def log(message):
    with PRINT_LOCK:
        print(message, flush=True)

def looks_html(content_type, head):
    head = head.lstrip().lower()
    return "text/html" in content_type.lower() or head.startswith((b"<!doctype html", b"<html", b"<head"))

def fetch(url, dest, optional=False, max_tries=4):
    """Download with an atomic .part file and HTTP Range-based resume."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return "exists", None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(max_tries):
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "MovingDrone20-downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with OPENER.open(req, timeout=90) as response:
                status = response.status
                if status not in (200, 206):
                    raise RuntimeError(f"HTTP {status}")
                mode = "ab" if offset and status == 206 else "wb"
                first = response.read(8192)
                if looks_html(response.headers.get("Content-Type", ""), first):
                    raise RuntimeError("server returned an HTML page rather than file data")
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected and expected.isdigit() else None
                count = len(first)
                with tmp.open(mode) as f:
                    f.write(first)
                    while True:
                        chunk = response.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        count += len(chunk)
                if expected is not None and count != expected:
                    raise IOError(f"truncated transfer: got {count}, expected {expected} bytes")
            os.replace(tmp, dest)
            return "downloaded", None
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:  # Partial file may already be complete
                tmp.unlink(missing_ok=True)
                continue
            if exc.code in (403, 404, 410):
                return ("missing" if optional or exc.code == 404 else "failed"), f"HTTP {exc.code}"
            error = f"HTTP {exc.code}: {exc.reason}"
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            error = str(exc)
            if "HTML page" in error:
                return ("missing" if optional else "failed"), error
        if attempt < max_tries - 1:
            time.sleep(min(2 ** attempt, 8))
    return "failed", error

def url_for(scene, rel):
    return f"{BASE}/sequences/{urllib.parse.quote(scene)}/{urllib.parse.quote(rel, safe='/')}"

def add(tasks, rel, optional=False):
    if rel and rel not in {x[0] for x in tasks}:
        tasks.append((rel, optional))

def get_jobs(meta, include_optional=True, include_normals=True, include_depth=False):
    tasks = []
    for name in ("poses.csv", "intrinsics.json", "video.mp4", "dsm.npz"):
        add(tasks, name)
    # Download ALL DOP years (not just the latest): file paths are provided by meta.json.
    dops = meta.get("dops", {})
    if dops:
        for entry in dops.values():
            if entry.get("file"):
                add(tasks, entry["file"])
            elif entry.get("year"):
                add(tasks, f"dop_{entry['year']}.jpg", True)
                add(tasks, "dop.jpg", True)
    else:
        if meta.get("dop", {}).get("file"):
            add(tasks, meta["dop"]["file"], True)
        else:
            add(tasks, "dop.jpg", True)
    if include_optional:
        for name in ("lod1.npz", "lod2.npz", "lidar.npz", "mesh.npz"):
            add(tasks, name, True)
    n = int(meta.get("num_frames", 0))
    for i in range(n):
        if include_depth:
            add(tasks, f"depth/depth_{i:04d}.png", True)
            add(tasks, f"depth/depth_{i:04d}.json", True)
        if include_normals:
            add(tasks, f"normals/normal_{i:04d}.npz", True)
    return tasks

def probe_normal_availability(scene, seq_dir):
    """Avoid thousands of HTTP 404s for newer scenes without stored normals."""
    rel = "normals/normal_0000.npz"
    status, err = fetch(url_for(scene, rel), seq_dir / rel, optional=True, max_tries=2)
    if status in ("downloaded", "exists"):
        return True
    log(f"  {scene}: per-frame normals not available (or inaccessible); skipping normals/*.npz")
    return False

def download_scene(scene, root, jobs, include_optional, include_normals, include_depth=False):
    seq_dir = root / "scenes" / scene
    meta_path = seq_dir / "meta.json"
    log(f"\n========== {scene} ==========")
    status, err = fetch(url_for(scene, "meta.json"), meta_path)
    if status == "failed" or not meta_path.exists():
        log(f"  ERROR meta.json: {err}")
        return [(scene, "meta.json", err or "missing")]
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        log(f"  ERROR invalid meta.json: {exc}")
        return [(scene, "meta.json", f"invalid JSON: {exc}")]
    available_normals = include_normals and probe_normal_availability(scene, seq_dir)
    tasks = get_jobs(meta, include_optional, available_normals, include_depth)
    total = len(tasks)
    failures = []
    counts = {"exists": 0, "downloaded": 0, "missing": 0, "failed": 0}
    def do_one(job):
        rel, optional = job
        state, error = fetch(url_for(scene, rel), seq_dir / rel, optional=optional)
        return rel, optional, state, error
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for index, (rel, optional, state, error) in enumerate(pool.map(do_one, tasks), 1):
            counts[state] += 1
            if state == "failed" or (state == "missing" and not optional):
                failures.append((scene, rel, error or state))
                log(f"  FAIL: {rel}: {error}")
            if index % 100 == 0 or index == total:
                log(f"  {index}/{total} files | new={counts['downloaded']} cached={counts['exists']} "
                    f"optional_unavailable={counts['missing']} failed={counts['failed']}")
    # If the published scene uses JPEG frames in place of video, retrieve all frame files.
    if not (seq_dir / "video.mp4").exists() and meta.get("num_frames", 0):
        log("  video.mp4 absent; downloading legacy per-frame JPEGs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            fallback = [(f"rendering/frame_{i:04d}.jpg", False) for i in range(int(meta['num_frames']))]
            for rel, optional, state, error in pool.map(do_one, fallback):
                if state not in ("exists", "downloaded"):
                    failures.append((scene, rel, error or state))
    return failures

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/Data/RIght/MovingDrone_20"))
    parser.add_argument("--jobs", type=int, default=4, help="max parallel transfers per scene, default 4")
    parser.add_argument("--dry-run", action="store_true", help="show selections, do not download")
    parser.add_argument("--skip-normals", action="store_true", help="skip optional normal archives")
    parser.add_argument("--with-depth", action="store_true", help="also download depth PNG/JSON (disabled by default)")
    parser.add_argument("--skip-mesh-lidar", action="store_true", help="skip optional mesh/LoD/LiDAR archives")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        parser.error("--jobs must be between 1 and 16")
    log(f"Selected {len(SCENES)} MovingDrone sequences:")
    for i, (kind, name, split) in enumerate(SCENES, 1):
        log(f"  {i:2d}. {name:27s} category={kind:23s} split={split}")
    log(f"Destination: {args.root}")
    log(f"Depth downloads: {'enabled' if args.with_depth else 'disabled'}; existing completed files will be skipped")
    if args.dry_run:
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "selection.json").write_text(
        json.dumps([dict(category=k, sequence=s, official_split=t) for k,s,t in SCENES], ensure_ascii=False, indent=2)
    )
    split_status, split_err = fetch(f"{BASE}/splits.json", args.root / "splits.json")
    if split_status == "failed":
        log(f"WARNING: splits.json: {split_err}")
    failures = []
    for _, name, _ in SCENES:
        failures.extend(download_scene(name, args.root, args.jobs,
                                       not args.skip_mesh_lidar, not args.skip_normals,
                                       args.with_depth))
    fail_path = args.root / "failed_downloads.tsv"
    with fail_path.open("w") as f:
        f.write("sequence\tpath\terror\n")
        for seq, rel, err in failures:
            f.write(f"{seq}\t{rel}\t{err}\n")
    log(f"\nCompleted attempt for 20 sequences. Failures: {len(failures)}")
    log(f"Missing OPTIONAL files are logged per scene, but do not count as failures.")
    log(f"Failure report: {fail_path}")
    log("Re-run the same command to retry failures and resume partial files.")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
