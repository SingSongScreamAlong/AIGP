# AI Grand Prix - Setup

## Dependencies

### PX4 Autopilot (SITL Simulator)
This project uses PX4 SITL for drone simulation. Clone it separately:

```bash
git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
cd ~/PX4-Autopilot
make px4_sitl_default
```

### Python Environment
```bash
cd ~/ai-grand-prix
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notes
- Scripts expect PX4-Autopilot at `~/PX4-Autopilot`
- The PX4 SITL sim writes to `/private/tmp/px4_sih_out.log` — this file can grow to hundreds of GB if left running. Always kill PX4 processes when done: `pkill -9 -f px4_sitl_default`
