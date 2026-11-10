# MovingDrone Tile Pipeline

This pipeline turns a manually selected DOP area and a hand-drawn 2D route into a rendered UAV sequence. It reuses the reference scene's mesh, DSM, camera calibration, and map data.

## Requirements

- Run from the OrthoTrack repository root in its Python environment.
- Open3D GPU rendering with EGL support is required for generation.
- Minimal reference assets are stored under `/mnt/Data/RIght/MovingDrone_Ref`.

## 1. Start the route editor

```bash
python dataset_pipline/route_editor/app.py --host 0.0.0.0 --port 8765
```

The defaults use `MovingDrone_Ref` for both reference and save roots. The editor reads scene folders from that root and saves each tile inside the matching scene directory, without duplicating the scene name: `MovingDrone_Ref/airport10/tile01/`, then `tile02/`, and so on. Open [http://localhost:8765](http://localhost:8765), select a scene, drag on the full reference DOP to choose a crop, and click **Confirm crop**. Select the new tile, draw a route, review the smoothed blue preview, and click **Save route**.

To work with one scene and set both paths explicitly:

```bash
python dataset_pipline/route_editor/app.py \
  --ref-path /mnt/Data/RIght/MovingDrone_Ref/airport10 \
  --save-path /mnt/Data/RIght/MovingDrone_Ref/airport10
```

To save tiles elsewhere, pass a different `--save-path`, such as `/mnt/Data/RIght/MovingDrone_Tiles`.

## 2. Plan and render a tile

```bash
python dataset_pipline/pipeline.py plan \
  --config /mnt/Data/RIght/MovingDrone_Ref/<scene>/tileXX/config.yaml

python dataset_pipline/pipeline.py generate \
  --config /mnt/Data/RIght/MovingDrone_Ref/<scene>/tileXX/config.yaml \
  --depth
```

Planning checks route speed, collision clearance, camera-view coverage, and map bounds. Generation writes the RGB video and, with `--depth`, one metric depth PNG per frame.

## 3. Validate the output

```bash
python dataset_pipline/pipeline.py validate \
  --config /mnt/Data/RIght/MovingDrone_Ref/<scene>/tileXX/config.yaml \
  --depth
```

Outputs are written to the tile directory: `video.mp4`, `depth/`, `poses.csv`, `intrinsics.json`, `meta.json`, `trajectory.json`, `dop/`, and `preview/`.

## Camera and map settings

Edit the tile's `config.yaml` before planning. `height_agl_m` is the target height above local DSM terrain. Camera tilt is measured from nadir: larger values look farther forward. The planner accepts **20°–30°** to avoid near-satellite views. `fov_vertical_deg` sets the vertical field of view.

The final DOP crop follows the UAV camera view and may be larger than the hand-selected route crop. It expands up to `max_view_crop_scale` (default `5.0`). The view must still fit the available DOP, DSM, and mesh; planning stops if coverage is insufficient. Move the crop or route inward, or use a compatible height and camera angle, then plan again.
