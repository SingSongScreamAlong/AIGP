"""Record a full drone race as an MP4 video."""
from pathlib import Path
import numpy as np
import mujoco
import imageio
from lsy_drone_racing.utils import load_config, load_controller
import gymnasium

config_path = Path.home() / "ai-grand-prix/sims/lsy_drone_racing/config/level0.toml"
config = load_config(config_path)
config.sim.render = False

control_path = Path.home() / "ai-grand-prix/sims/lsy_drone_racing/lsy_drone_racing/control"
controller_path = control_path / config.controller.file
controller_cls = load_controller(controller_path)

env = gymnasium.make(
    config.env.id, freq=config.env.freq, sim_config=config.sim,
    sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
    track=config.env.track,
)

obs, info = env.reset()
controller = controller_cls(obs, info, config)
sim = env.unwrapped.sim

renderer = mujoco.Renderer(sim.mj_model, height=480, width=640)
frames = []
render_every = max(1, config.env.freq // 30)

print(f"Running race at {config.env.freq}Hz, capturing every {render_every} steps (~30fps)...")

i = 0
terminated = False
truncated = False

while not terminated and not truncated:
    action = controller.compute_control(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    i += 1
    if i % render_every == 0:
        renderer.update_scene(sim.mj_data)
        frame = renderer.render()
        frames.append(frame.copy())

gates = obs["target_gate"]
if gates == -1:
    gates = len(config.env.track.gates)
print(f"Race done! Gates: {gates}/{len(config.env.track.gates)}, Steps: {i}, Frames: {len(frames)}")

out_path = str(Path.home() / "ai-grand-prix/race_replay.mp4")
imageio.mimsave(out_path, frames, fps=30)
print(f"Video saved: {out_path}")

renderer.close()
env.close()
