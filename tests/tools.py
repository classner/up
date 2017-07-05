"""Tool unittests."""
# pylint: disable=invalid-name
import os
import os.path as path
import unittest
import numpy as np


class MeshTest(unittest.TestCase):

    """Test Mesh serialization."""

    def test_constructor(self):
        """Test object construction."""
        import up_tools.mesh as upm
        # From file.
        tm = upm.Mesh(filename=path.join(path.dirname(__file__),
                                         '..',
                                         'models',
                                         '3D',
                                         'template-bodyparts.ply'))
        # From data.
        v = np.zeros((3, 3))
        vc = np.zeros((3, 3))
        f = np.zeros((1, 3))
        tm = upm.Mesh(v=v, vc=vc, f=f)
        self.assertTrue(np.all(v == tm.v))
        self.assertTrue(np.all(vc == tm.vc))
        self.assertTrue(np.all(f == tm.f))

    def test_serialization(self):
        """Test serialization."""
        import up_tools.mesh as upm
        tm = upm.Mesh(filename=path.join(path.dirname(__file__),
                                         '..',
                                         'models',
                                         '3D',
                                         'template-bodyparts.ply'))
        self.assertTrue(tm.v is not None)
        self.assertTrue(tm.vc is not None)
        self.assertTrue(tm.f is not None)
        tm.write_ply('test_out.ply')
        rm = upm.Mesh(filename='test_out.ply')
        self.assertTrue(np.all(tm.v == rm.v))
        self.assertTrue(np.all(tm.vc == rm.vc))
        self.assertTrue(np.all(tm.f == rm.f))
        os.remove('test_out.ply')


if __name__ == '__main__':
    unittest.main()
