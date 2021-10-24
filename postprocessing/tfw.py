from dataclasses import dataclass
from typing import Tuple


@dataclass
class TFWCoordinates:
    x_pixel_size: float
    y_rot: float
    x_rot: float
    y_pixel_size: float
    x: float
    y: float


def read_tfw_file(path: str) -> TFWCoordinates:
    with open(path, "rt") as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.strip(), lines))
        lines = lines[:6]
        assert len(lines) == 6, "Not enough lines in tfw file"

        x_pixel_size = float(lines[0])
        y_rot = float(lines[1])
        x_rot = float(lines[2])
        y_pixel_size = float(lines[3])
        x = float(lines[4])
        y = float(lines[5])

        return TFWCoordinates(
            x_pixel_size=x_pixel_size,
            y_rot=y_rot,
            x_rot=x_rot,
            y_pixel_size=y_pixel_size,
            x=x,
            y=y,
        )


def pixel_coord_to_world_coords(tfw: TFWCoordinates, x: float, y: float) -> Tuple[float, float]:
    if not tfw.y_rot == tfw.x_rot == 0:
        raise NotImplementedError

    x_world_coord = tfw.x + x * tfw.x_pixel_size
    y_world_coord = tfw.y - y * tfw.x_pixel_size

    return x_world_coord, y_world_coord


if __name__ == "__main__":
    tfw = read_tfw_file("3560-825.tfw")
    print(tfw)
    print(pixel_coord_to_world_coords(tfw, 5000, 5000))
