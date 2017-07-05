#!/usr/bin/env python2
"""Tests for the conversion code."""
import unittest

import numpy as np
import pyximport; pyximport.install()  # pylint: disable=multiple-statements
import conversions  # pylint: disable=import-error


class ConversionTests(unittest.TestCase):

    """Tests for the conversions."""

    def _create_test_axis_angle_matrix(self):  # pylint: disable=no-self-use
        """Create a test matrix."""
        angles = np.random.normal(size=(1000, 3)).astype('float32')
        for angle_idx, angle_w in enumerate(angles):
            angle_w = angles[angle_idx]
            angle_wnorm = np.linalg.norm(angle_w)
            if angle_wnorm > np.pi:
                angles[angle_idx] = angle_w / angle_wnorm * np.fmod(angle_wnorm, np.pi)
        return angles

    def test_axis_angle_versor(self):
        """Test cycling from axis angle through versor back."""
        np.random.seed(1)
        angles = self._create_test_axis_angle_matrix()
        vers = conversions.axis_angle_to_versor_ipr(angles.copy())
        angles_res = conversions.versor_to_axis_angle_ipr(vers)
        '''
        for idx in range(1000):
            if not np.allclose(angles[idx], angles_res[idx]):
                print idx, angles[idx], angles_res[idx]
        '''
        self.assertTrue(np.allclose(angles, angles_res, atol=1e-5, rtol=1e-4))

    def test_matrix_to_versor(self):
        """Test matrix to versor conversion."""
        np.random.seed(1)
        angles = self._create_test_axis_angle_matrix()
        matvals = np.array([-0.40967655,  0.07308591, -0.90929848, 0.07226962,
                   -0.99105203, -0.1122174, -0.90936369, -0.11168748,
                   0.40072888], dtype=np.float32)
        vers = conversions.matrix_to_versor(matvals.reshape((1, 9)))
        aangle = conversions.versor_to_axis_angle_ipr(vers)

    def test_matrix_cycle(self):
        """Test cycling from axis angle through matrix through versor back."""
        np.random.seed(1)
        angles = self._create_test_axis_angle_matrix()
        mat = conversions.axis_angle_to_matrix(angles)
        vers = conversions.matrix_to_versor(mat)
        angles_res = conversions.versor_to_axis_angle_ipr(vers)
        '''
        for idx in range(1000):
            if not np.allclose(angles[idx], angles_res[idx]):
                print idx
        '''
        self.assertTrue(np.allclose(angles, angles_res))

    def test_matrix_projection(self):
        """Test the manifold projection."""
        np.random.seed(1)
        angles = self._create_test_axis_angle_matrix()
        mat = conversions.axis_angle_to_matrix(angles)
        n_failed = 0
        for matrix_id, matrix in enumerate(mat):  # pylint: disable=unused-variable
            self.assertTrue(np.allclose(np.linalg.det(matrix.reshape((3, 3))),
                                        1.))
            """
            if not np.allclose(np.dot(matrix.reshape((3, 3)),
                                      matrix.reshape((3, 3)).T),
                               np.eye(3), atol=1e-6):
                print np.dot(matrix.reshape((3, 3)), matrix.reshape((3, 3)).T), matrix_id
            """
            self.assertTrue(np.allclose(np.dot(matrix.reshape((3, 3)),
                                               matrix.reshape((3, 3)).T),
                                        np.eye(3), atol=1e-6))
            # Perturb.
            offs = np.random.normal()
            idx = np.random.randint(low=0, high=9)
            matrix[idx] += offs
            # Recover.
            try:
                rec, dist = conversions.project_to_rot(matrix)  # pylint: disable=unused-variable
                self.assertTrue(np.allclose(np.linalg.det(rec.reshape((3, 3))),
                                            1., atol=1e-5))
                """
                if not np.allclose(np.dot(rec.reshape((3, 3)),
                                          rec.reshape((3, 3)).T),
                                   np.eye(3), atol=1e-6):
                    print np.dot(rec.reshape((3, 3)),
                                 rec.reshape((3, 3)).T), matrix_id
                """
                self.assertTrue(np.allclose(np.dot(rec.reshape((3, 3)),
                                                   rec.reshape((3, 3)).T),
                                            np.eye(3), atol=1e-6))
            except:  # pylint: disable=bare-except
                n_failed += 1
        self.assertEqual(n_failed, 56)
        # print 'could not recover matrix in %d cases.' % (n_failed)


class RotationForestTestCase(unittest.TestCase):

    """Test the rotation forests."""

    def _create_test_axis_angle_matrix(self, n_angles=1):  # pylint: disable=no-self-use
        """Create a test matrix."""
        angles = np.random.normal(size=(1000, 3*n_angles)).astype('float32')
        for angle_idx, angle_w in enumerate(angles):
            for line_idx in range((angles.shape[1] // 3)):
                angle_w = angles[angle_idx, line_idx * 3]
                angle_wnorm = np.linalg.norm(angle_w)
                if angle_wnorm > np.pi:
                    angles[angle_idx, line_idx * 3:(line_idx+1) * 3] = \
                        angle_w / angle_wnorm * np.fmod(angle_wnorm, np.pi)
        return angles

    def test_basic(self):
        """Test the basic functionality."""
        #np.seterr(all='raise')
        angles = self._create_test_axis_angle_matrix()
        mat = conversions.axis_angle_to_matrix(angles)
        dta = np.random.uniform(size=(1000, 5))
        from rotation_forest import RotationForest
        rf = RotationForest(n_jobs=2)  # pylint: disable=invalid-name
        rf.fit(dta, mat)
        res = rf.predict(dta)
        import cPickle as pickle
        pstr = pickle.dumps(rf)
        nf = pickle.loads(pstr)  # pylint: disable=invalid-name
        self.assertTrue(np.all(nf.predict(dta) == res))

    def test_multitarget(self):
        """Test multiple prediction target functionality."""
        #np.seterr(all='raise')
        angles = self._create_test_axis_angle_matrix(n_angles=2)
        mat = conversions.axis_angle_to_matrix(angles)
        dta = np.random.uniform(size=(1000, 5))
        from rotation_forest import RotationForest
        rf = RotationForest(n_jobs=2)  # pylint: disable=invalid-name
        rf.fit(dta, mat)
        res = rf.predict(dta)
        import cPickle as pickle
        pstr = pickle.dumps(rf)
        nf = pickle.loads(pstr)  # pylint: disable=invalid-name
        self.assertTrue(np.all(nf.predict(dta) == res))


    def test_opencv_rodrigues(self):
        """Test the manifold projection."""
        import cv2
        np.random.seed(1)
        angles = self._create_test_axis_angle_matrix()
        mat = conversions.axis_angle_to_matrix(angles)
        n_failed = 0
        for matrix_id, matrix in enumerate(mat):  # pylint: disable=unused-variable
            self.assertTrue(np.allclose(np.linalg.det(matrix.reshape((3, 3))),
                                        1.))
            """
            if not np.allclose(np.dot(matrix.reshape((3, 3)),
                                      matrix.reshape((3, 3)).T),
                               np.eye(3), atol=1e-6):
                print np.dot(matrix.reshape((3, 3)), matrix.reshape((3, 3)).T), matrix_id
            """
            self.assertTrue(np.allclose(np.dot(matrix.reshape((3, 3)),
                                               matrix.reshape((3, 3)).T),
                                        np.eye(3), atol=1e-6))
            # Perturb.
            offs = np.random.normal()
            idx = np.random.randint(low=0, high=9)
            matrix[idx] += offs
            # Recover.
            rec = cv2.Rodrigues(cv2.Rodrigues(matrix.reshape((3, 3)))[0])[0]  # pylint: disable=unused-variable
            self.assertTrue(np.allclose(np.linalg.det(rec.reshape((3, 3))),
                                        1., atol=1e-5))
            """
            if not np.allclose(np.dot(rec.reshape((3, 3)),
            rec.reshape((3, 3)).T),
            np.eye(3), atol=1e-6):
            print np.dot(rec.reshape((3, 3)),
            rec.reshape((3, 3)).T), matrix_id
            """
            self.assertTrue(np.allclose(np.dot(rec.reshape((3, 3)),
                                               rec.reshape((3, 3)).T),
                                        np.eye(3), atol=1e-6))
        # print 'could not recover matrix in %d cases.' % (n_failed)


if __name__ == '__main__':
    unittest.main()
