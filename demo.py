"""One synthetic pass, distances in km. Run with Python 3."""
from pathlib import Path
from math import hypot

USER = (10, 10)
RECEIVER = (-5, 0)
RADIUS = 50


def covered(point, center):
    return hypot(point[0] - center[0], point[1] - center[1]) <= RADIUS


def receive_messages():
    # One downlink per step if the user can be served. The receiver listens
    # to the same beam. All messages carry the same observable session ID.
    for step, x in enumerate(range(-90, 91, 15)):
        beam = (x, 0)
        if covered(USER, beam) and covered(RECEIVER, beam):
            yield step, beam


def estimate(observations):
    cells = {(x, y) for x in range(-150, 151) for y in range(-60, 61)}
    for step, beam in observations:
        cells = {p for p in cells if covered(p, beam)}
        yield step, beam, cells


def main():
    observations = list(receive_messages())
    layers = []
    for step, beam, cells in estimate(observations):
        print(f"step={step:2}  area~{len(cells):5} km²  contains_user={USER in cells}")
        layers.append(f'<circle cx="{beam[0]}" cy="{beam[1]}" r="{RADIUS}" fill="none" stroke="#7097b3" stroke-width=".7"/>')
    if not observations:
        print("No observed downlink.")
        return
    # Each retained grid point represents one square kilometre.
    shape = " ".join(f"M{x-.5},{y-.5}h1v1h-1z" for x, y in sorted(cells))
    layers.insert(0, f'<path d="{shape}" fill="#87cfb0"/>')
    for point, color, label in [(USER, "#ae322d", "user"), (RECEIVER, "#182e55", "receiver")]:
        x, y = point
        layers.append(f'<circle cx="{x}" cy="{y}" r="1.7" fill="{color}"/><text x="{x+3}" y="{y}" font-size="4">{label}</text>')
    out = Path(__file__).with_name("result.svg")
    out.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="-105 -55 210 110">' + "".join(layers) + '</svg>', encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
