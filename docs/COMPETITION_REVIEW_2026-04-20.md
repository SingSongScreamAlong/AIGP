# AI Grand Prix — Competition Review
**Date:** 2026-04-20
**Sources:** theaigrandprix.com / dcl-project.com content as quoted by DCL, Anduril newsroom, DroneLife, DroneXL, TechCrunch, Fortune, OhioTechNews, Built In, Palmer Luckey/Anduril on X. Direct access to theaigrandprix.com is blocked by the network egress proxy in this environment; this review is a reconstruction from indexed and republished content.

## 1. Identity, Founders, Partners

- **Competition name:** The AI Grand Prix — Season 1
- **Founder/host:** Anduril Industries; concept attributed to Anduril founder Palmer Luckey
- **Partners:** Drone Champions League (DCL), Neros Technologies, JobsOhio
- **Format:** Fully autonomous drone racing — identical Neros drones, no human pilots, no hardware modifications, software is the only lever
- **Scale:** Over 1,000 teams signed up within 24 hours of the Jan 27, 2026 announcement (per Palmer Luckey on X)

## 2. Prize Structure

- **Total prize pool:** $500,000
- **Top 10 at the Ohio final:** guaranteed cash prize of at least $5,000
- **Job prize:** Highest-scoring participant (or a single eligible member of the highest-scoring team) can bypass Anduril's standard recruiting cycle and interview directly with hiring managers for relevant open roles
- **Registration fee:** None

## 3. Timeline (best reconstructed)

| Phase | Window | Notes |
|---|---|---|
| Interface spec release | Second half of March 2026 | VADR-TS-001 matches this — already in hand |
| Simulator + course release | May 2026 | DCL-built, Python-based |
| Virtual qualification — Round 1 | May – July 2026 (some sources say April–June) | Time-trial on simple desaturated course with highlighted gates |
| Virtual qualification — Round 2 | Cutoff ~end of July 2026 | 3D-scanned realistic environment, visually complex |
| Physical qualifier | September 2026, Southern California | Two-week in-person training and qualification with the real drone |
| Final — AI Grand Prix Ohio | November 2026, Columbus | Head-to-head at Anduril's Arsenal-1 facility |

**Date ambiguity:** Sources split between "April–June" and "May–July" for the virtual phase. Given the stated "simulator releases in May," May–July is more consistent.

## 4. Eligibility & Registration

- **Team size:** Individuals, or teams of up to 8 members
- **Restriction:** One team per person; participating on more than one team (or individually AND on a team) is grounds for DQ
- **Ineligible:** Employees of Anduril, DCL, or Neros cannot participate or win
- **Geographic:** Citizens of the Russian Federation prohibited as participants or spectators
- **Proof of citizenship:** Required prior to in-person qualifier
- **Minors:**
  - 14–17 may self-register but must submit parent/guardian contact
  - Under 14: parent/guardian must register on their behalf
  - Written parental consent + age verification required for all minors
- **Open to:** Individuals, university teams, research organizations — no professional credentials required

## 5. Hardware (identical for all competitors)

- **Drone:** Neros-built, likely an 8-inch quadcopter variant (Neros Archer family: 5", 8", 10", and a fiber-optic variant exist; AI Grand Prix uses one spec)
- **Camera:** Single FPV camera, ~12MP wide-angle, monocular only
- **No LiDAR:** Explicitly called out — "challenge is to find a clear path without reliable depth information"
- **Compute:** DCL's "AI vector module" — described as more capable than a Raspberry Pi, in the ~100 TOPS range (exact spec not yet published). Consistent with Jetson Orin NX class.
- **IMU:** accelerometer, gyroscope, and likely motor RPM readouts
- **Other:** Onboard AI compute with RAM/SSD, camera feed, Wi-Fi/Bluetooth (per module description)
- **Hardware mods:** Prohibited

## 6. Software Stack (virtual phase)

- **Language:** Python is the primary interface
- **Compiled extensions:** C / Cython and similar generally expected to be possible
- **Libraries:** Teams can "broadly use the libraries they need"; restrictions to be communicated later
- **Connectivity to pilot the drone:** DQ

## 7. Environment & Sensing

- **Virtual track:** Indoor-style, consistent lighting for fairness
- **Round 1 course:** Intentionally simple — small number of gates, desaturated environment, visually highlighted gates to help teams get started
- **Round 2 course:** Significantly more realistic and visually complex — real 3D-scanned environment, obstacles, and visual "distractions"
- **Physical course:** Indoor, consistent lighting, obstacles, visual distractions
- **Position data:** Teams may receive limited coordinate information for the **starting position** in the virtual qualifier. Beyond that, fly **without coordinate/position data** — pure visual/IMU
- **Scoring:** Time-trial. Everyone runs the same course. Fastest valid times advance. Runs must successfully pass gates to count. Speed vs reliability trade-off is explicit.

## 8. Physical Qualifier (September 2026, SoCal)

- Two-week in-person training + qualification experience
- Teams refine software code on the real drone to achieve fastest real-course time
- Standardized UAS provided by AI Grand Prix
- Bringing the competition "from simulation to the real world"

## 9. Final (November 2026, Columbus)

- **Venue:** Arsenal-1 — Anduril's planned 5-million-sq-ft manufacturing facility outside Columbus, OH
- **Format:** Live, head-to-head autonomous drone race
- **Hosts:** Anduril + JobsOhio

## 10. Communications

- Updates, parameters, and previews via **weekly newsletters** and the site **FAQ**
- Announcement channels: theaigrandprix.com, dcl-project.com (mirror), Anduril newsroom, Anduril + Palmer Luckey on X

## 11. What I could NOT extract

- Exact simulator SDK / API signature (not yet released publicly; due May)
- Exact compute module part number (TBD)
- Exact camera frame rate, FOV, intrinsics (TBD)
- Verbatim "Previous Updates" post list — the page is blocked by egress; I could only reconstruct themes from republished content
- Scoring formula details (time-based is confirmed; any bonuses/penalties/tie-breakers TBD)
- Prize distribution curve beyond "Top 10 ≥ $5k"

## 12. Delta vs our existing STATUS doc

**Confirms / aligns with what we already have:**
- VADR-TS-001 matches the "interface spec in H2 March" promise
- Monocular FPV, no LiDAR, identical hardware
- Python stack, virtual → physical → final structure

**New information for planning:**
1. **Round 1 is explicitly easy** (desaturated, highlighted gates). Our VirtualCamera-based stack may be adequate for Round 1; the Round 2 "real 3D-scanned environment + distractors" is where YOLO+real-image training becomes necessary.
2. **Obstacles / visual distractors** are called out for both Round 2 and the physical track. Gate-following alone isn't enough — some distractor robustness testing belongs in the harness.
3. **No coordinate data beyond starting pose.** This confirms that full visual-inertial estimation is required — not a nice-to-have. The gate belief model (and the yaw bug fix) matter.
4. **Compiled extensions are allowed.** YOLO in ONNX/TensorRT, C extensions for PnP, etc. are on the table.
5. **Indoor consistent lighting at the physical qualifier.** Lighting augmentation matters less than I'd assumed; sim-to-real gate-geometry gap matters more.
6. **Time-trial with gate-validity requirement.** Optimize completion rate first, then time. A 95% completer at 28 s beats a 60% completer at 22 s if the median decides cutoff.
7. **Two-week SoCal window is a hard compute/travel constraint.** Whatever pipeline makes it through virtual Round 2 needs to be deployable on hardware we don't touch until Sept.
8. **~1000+ teams.** Competitive field — top-1% of 1000 is still ~10 teams making it to the physical qualifier (exact number TBD).

## 13. Recommended adjustments to the project plan

- **P0 belief fix still dominant.** The "no coordinate data" confirmation makes the gate belief propagation bug a must-fix before May, since that's the only thing holding gate estimates together across dropouts.
- **P1 now has two branches:** (a) wire real YOLO into the stack (as planned) and (b) build a distractor-augmented training set — NOT just clean gates.
- **Round-1 vs Round-2 planning.** Two-stage goal: pass Round 1 on the VirtualCamera simulation adapter with minimal rework when the DCL sim drops; then harden for Round 2's 3D-scanned course.
- **Completion-rate telemetry.** Add a completion-rate KPI to the A/B harness alongside lap time. Our current scoring has been implicit about this; make it explicit.
- **DCL sim adapter is now P1.** Simulator drops in May. Scaffolding an adapter that routes the DCL Python API to the existing planner/estimator is the single highest-leverage item for the next month.
