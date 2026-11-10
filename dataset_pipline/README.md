# MovingDrone Tile Pipeline

Create UAV sequences from a hand-cropped DOP tile and a drawn 2D route. Rendering reuses the reference scene's mesh and DOP.

## Start the editor

From the OrthoTrack repository root:

```bash
python dataset_pipline/route_editor/app.py \
  --ref-path /path/to/MovingDrone_Ref \
  --host 0.0.0.0 --port 8765
```

Open [http://localhost:8765](http://localhost:8765). Tiles are saved under the reference path by default. Pass `--save-path /path/to/MovingDrone_Tiles` to save them elsewhere. A tile is stored as `<save-path>/<scene>/tile01/`.

## Create and render a tile

1. In **Draw route**, select a scene, crop its DOP, and draw a route.
2. Set **Height AGL**, **Tilt from nadir**, **Frames**, and **Camera aim**. The previews show the camera view, mesh height, route speed, and camera direction.
3. Click **Plan + Preview first RGB**. Adjust the route or frame count if planning fails. More frames give the UAV more time to travel the same route, reducing its speed.
4. In **Generate data**, select the saved tile and choose **RGB video**, **Depth maps**, or both.

Tiles are saved under their scene, for example:

```text
MovingDrone_Tiles/airport10/tile01/
```

The editor's default camera tilt range is 20°–30° from nadir. Generation requires the OrthoTrack GPU rendering environment with Open3D/EGL.

## Command line

```bash
python dataset_pipline/pipeline.py plan --config /path/to/tile/config.yaml
python dataset_pipline/pipeline.py preview --config /path/to/tile/config.yaml
python dataset_pipline/pipeline.py generate --config /path/to/tile/config.yaml --depth
python dataset_pipline/pipeline.py validate --config /path/to/tile/config.yaml --depth
```

Omit `--depth` to generate RGB only. Add `--no-rgb --depth` for depth only. Outputs are stored in the tile directory, including `video.mp4` when RGB is selected and `depth/` when depth is selected.
