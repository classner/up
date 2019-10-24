"""Mesh tools."""
# pylint: disable=invalid-name
import numpy as np
import meshio


class Mesh(object):  # pylint: disable=too-few-public-methods

    """An easy to use mesh interface."""

    def __init__(self, filename=None, v=None, vc=None, f=None):
        """Construct a mesh either from a file or with the provided data."""
        if filename is not None:
            assert v is None and f is None and vc is None
        else:
            assert v is not None or f is not None
            if vc is not None:
                assert len(v) == len(vc)

        if filename is None:
            self.v = v
            self.vc = vc
            self.f = f
        else:
            mesh = meshio.read(filename)
            self.v = mesh.points
            self.vc = (
                np.column_stack(
                    [
                        mesh.point_data["red"],
                        mesh.point_data["green"],
                        mesh.point_data["blue"],
                    ]
                ).astype(np.float)
                / 255.0
            )
            self.f = mesh.cells["triangle"]

    def write_ply(self, out_name):
        """Write to a .ply file."""
        colors = (self.vc * 255).astype("uint8")
        meshio.write_points_cells(
            out_name,
            self.v,
            {"triangle": self.f},
            point_data={
                "red": colors[:, 0],
                "green": colors[:, 1],
                "blue": colors[:, 2],
            },
        )
