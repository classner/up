"""Mesh tools."""
# pylint: disable=invalid-name
import numpy as np
import numpy.lib.recfunctions as rfn
import random
import scipy.sparse as sp
import plyfile


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

        if filename is not None:
            plydata = plyfile.PlyData.read(filename)
            self.v = np.hstack((np.atleast_2d(plydata['vertex']['x']).T,
                                np.atleast_2d(plydata['vertex']['y']).T,
                                np.atleast_2d(plydata['vertex']['z']).T))
            self.vc = np.hstack((np.atleast_2d(plydata['vertex']['red']).T,
                                 np.atleast_2d(plydata['vertex']['green']).T,
                                 np.atleast_2d(plydata['vertex']['blue']).T)).astype('float') / 255.
            # Unfortunately, the vertex indices for the faces are stored in an
            # object array with arrays as objects. :-/ Work around this.
            self.f = np.vstack([np.atleast_2d(elem) for
                                elem in list(plydata['face']['vertex_indices'])]).astype('uint32')
        else:
            self.v = v
            self.vc = vc
            self.f = f

    def write_ply(self, out_name):
        """Write to a .ply file."""
        vertex = rfn.merge_arrays([
            self.v.view(dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),]),
            (self.vc * 255.).astype('uint8').view(
                dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
            ],
                                  flatten=True,
                                  usemask=False)
        face = self.f.view(dtype=[('vertex_indices', 'i4', (3,))])[:, 0]
        vert_el = plyfile.PlyElement.describe(vertex, 'vertex')
        face_el = plyfile.PlyElement.describe(face, 'face')
        plyfile.PlyData([
            vert_el,
            face_el
        ]).write(out_name)

