"""Draw the README's explanatory simulation-to-real workflow as a standalone SVG."""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/system_overview.svg"


def text(x, y, value, size=21, color="#344054", weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def box(x, y, number, heading, lines, color, fill):
    parts = [f'<rect x="{x}" y="{y}" width="466" height="172" rx="18" '
             f'fill="{fill}" stroke="{color}" stroke-width="1.7"/>',
             f'<circle cx="{x+39}" cy="{y+39}" r="20" fill="{color}"/>',
             text(x+39, y+46, str(number), 22, "#ffffff", 700, "middle"),
             text(x+73, y+46, heading, 25, "#172b4d", 700)]
    for i, line in enumerate(lines):
        parts.append(text(x+26, y+91+i*30, line))
    return "\n".join(parts)


def main():
    blue, blue_fill = "#2463a8", "#f5f9ff"
    green, green_fill = "#147d69", "#f3faf7"
    sim = [
        ("Build the simulated task", ["MuJoCo + robosuite tabletop cube task",
          "FR3v2 arm, gripper, table and cube", "Set the workspace and lift objective"]),
        ("Define the policy input", ["14 values: arm, gripper and cube state",
          "Express positions in the robot frame", "4 outputs: x/y/z motion + gripper"]),
        ("Train the SAC policy", ["Learn from movement, contact and reward",
          "Workspace limits constrain motion", "Rewards favor a small, stable lift"]),
        ("Evaluate and save", ["Measure reward and lift/hold success",
          "Save the final policy and experiment logs", "Final scheduled evaluation: 9/10 holds"]),
    ]
    real = [
        ("Locate the real cube", ["RealSense RGB images + ArUco marker",
          "Hand-eye calibration links the camera", "coordinates to the robot base"]),
        ("Build the real policy input", ["Read the robot pose from ROS TF",
          "Add the calibrated marker pose", "Use the same 14-value input layout"]),
        ("Run the saved policy", ["Load the simulation-trained SAC policy",
          "Predict the next approach movement", "Scale, filter and limit the target"]),
        ("Execute grasp and lift", ["MoveIt + FR3 controller move the arm",
          "Policy + proximity trigger closing", "A preset motion lifts vertically"]),
    ]
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1110" viewBox="0 0 1080 1110" role="img" aria-labelledby="title description">
<title id="title">Simulation training and real FR3 deployment workflow</title>
<desc id="description">Two columns with eight named steps: build the simulated task, define input, train SAC, evaluate and save; locate the real cube, build the real input, run the saved policy, execute grasp and lift. Dashed links connect the shared input layout and saved policy transfer.</desc>
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M1 1 L8 5 L1 9" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>
  <marker id="transfer-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L8 5 L1 9" fill="none" stroke="#936d22" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></marker>
</defs>
<rect width="1080" height="1110" fill="#ffffff"/>
<g font-family="Segoe UI, Arial, sans-serif">
<rect x="165" y="24" width="750" height="96" rx="20" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>''',
        text(540, 63, "TASK: GRASP A CUBE, LIFT IT, KEEP IT STEADY", 24, "#172b4d", 700, "middle"),
        text(540, 94, "Learn in simulation, then adapt the policy to a real Franka Research 3", 20, anchor="middle"),
        text(257, 167, "SIMULATION TRAINING", 25, blue, 700, "middle"),
        text(257, 195, "Learning + quantitative evaluation", 21, anchor="middle"),
        text(823, 167, "REAL ROBOT DEPLOYMENT", 25, green, 700, "middle"),
        text(823, 195, "Perception + policy-assisted execution", 21, anchor="middle"),
    ]
    for row in range(4):
        y = 220 + row*212
        parts.append(box(24, y, row+1, *sim[row], blue, blue_fill))
        parts.append(box(590, y, row+5, *real[row], green, green_fill))
        if row < 3:
            for x in (257, 823):
                parts.append(f'<path d="M{x} {y+175} V{y+205}" fill="none" stroke="#64748b" stroke-width="2.2" marker-end="url(#arrow)"/>')
    parts.extend([
        '<path d="M493 559 H587" fill="none" stroke="#64748b" stroke-width="2.1" stroke-dasharray="6 5" marker-start="url(#arrow)" marker-end="url(#arrow)"/>',
        text(540, 521, "Same", 18, "#526072", 600, "middle"),
        text(540, 545, "input layout", 18, "#526072", 600, "middle"),
        '<path d="M493 942 H521 V773 H587" fill="none" stroke="#936d22" stroke-width="2.1" stroke-dasharray="6 5" marker-end="url(#transfer-arrow)"/>',
        text(551, 853, "Saved", 18, "#805e1d", 600, "middle"),
        text(551, 877, "policy", 18, "#805e1d", 600, "middle"),
        text(540, 1071, "Solid arrows: workflow steps. Dashed arrows: shared input layout and policy transfer.", 19, anchor="middle"),
        '</g></svg>',
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(parts)+"\n", encoding="utf-8", newline="\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
