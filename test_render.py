from pathlib import Path
import numpy as np
from lsy_drone_racing.utils import load_config
import gymnasium

config_path = Path.home() / "ai-grand-prix/sims/lsy_drone_racing/config/level0.toml"
config = load_config(config_path)
config.sim.render = True

env = gymnasium.make(
    config.env.id, freq=config.env.freq, sim_config=config.sim,
    sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
    track=config.env.track,
)
obs, info = env.reset()
result = env.render()
print("Render returns:", type(result))

sim = env.unwrapped.sim
print("Sim type:", type(sim))
print("Has viewer:", hasattr(sim, 'viewer'), getattr(sim, 'viewer', None) is not None)

try:
    import mujoco
    renderer = mujoco.Renderer(sim.mj_model, height=480, width=640)
    renderer.update_scene(sim.mj_data)
    frame = renderer.render()
    print(f"Offscreen works! Shape: {frame.shape}")
    import imageio
    imageio.imwrite(str(Path.home() / 'ai-grand-prix/gate_frame.png'), frame)
    print("Saved gate_frame.png!")
    renderer.close()
except Exception as e:
    print(f"Offscreen failed: {e}")
env.close()
