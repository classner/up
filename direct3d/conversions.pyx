"""Angle representation conversion functions in Cython.

Useful reference:
http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/
"""
# cython: infer_types=True
# cython: boundscheck=True
# cython: wraparound=False
import numpy as np
from scipy.linalg import polar
cimport numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


cpdef axis_angle_to_matrix(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 3 == 0
    cdef int sample_idx, rot_idx
    cdef np.float32_t angle
    cdef np.ndarray[np.float32_t, ndim=1] rep
    cdef np.float32_t x, y, z, s, c, t
    cdef np.float32_t tmp1, tmp2
    cdef np.ndarray[np.float32_t, ndim=2] retarr
    cdef np.float32_t[:,:] workarr
    retarr = np.zeros((anglearr.shape[0], anglearr.shape[1] * 3), dtype=np.float32)
    for sample_idx in range(anglearr.shape[0]):
        for rot_idx in range(0, anglearr.shape[1], 3):
            rep = anglearr[sample_idx, rot_idx:rot_idx + 3]
            workarr = retarr[sample_idx, rot_idx // 3 * 9:(rot_idx // 3 + 1) * 9].reshape((3, 3))
            angle = np.linalg.norm(rep)
            if angle == 0.:
                x, y, z = 0., 0., 0.
            else:
                x, y, z = rep / angle
            s = np.sin(angle)
            c = np.cos(angle)
            t = 1. - c
            # Diagonal.
            workarr[0, 0] = c + x*x*t
            workarr[1, 1] = c + y*y*t
            workarr[2, 2] = c + z*z*t
            tmp1 = x*y*t
            tmp2 = z*s
            # Out of diagonal.
            workarr[1, 0] = tmp1 + tmp2
            workarr[0, 1] = tmp1 - tmp2
            tmp1 = x*z*t
            tmp2 = y*s
            workarr[2, 0] = tmp1 - tmp2
            workarr[0, 2] = tmp1 + tmp2
            tmp1 = y*z*t
            tmp2 = x*s
            workarr[2, 1] = tmp1 + tmp2
            workarr[1, 2] = tmp1 - tmp2
    return retarr


cpdef matrix_to_versor(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 9 == 0
    cdef np.float32_t tr
    cdef int sample_idx, rot_idx
    cdef np.float32_t S, w
    cdef np.ndarray[np.float32_t, ndim=2] retarr, rep
    cdef np.ndarray[np.float32_t, ndim=1] workarr
    retarr = np.zeros((anglearr.shape[0], anglearr.shape[1] // 3), dtype=np.float32)
    for sample_idx in range(anglearr.shape[0]):
        for rot_idx in range(0, anglearr.shape[1], 9):
            rep = anglearr[sample_idx, rot_idx:rot_idx + 9].reshape((3, 3))
            workarr = retarr[sample_idx, rot_idx // 9 * 3:(rot_idx // 9 + 1) * 3]
            # Determine the trace.
            tr = rep[0, 0] + rep[1, 1] + rep[2, 2]
            if tr > 0:
                S = np.sqrt(tr + 1.) * 2  # S=4*q
                w = 0.25 * S
                workarr[0] = (rep[2, 1] - rep[1, 2]) / S
                workarr[1] = (rep[0, 2] - rep[2, 0]) / S
                workarr[2] = (rep[1, 0] - rep[0, 1]) / S
            elif rep[0, 0] > rep[1, 1] and rep[0, 0] > rep[2, 2]:
                S = np.sqrt(1. + rep[0, 0] - rep[1, 1] - rep[2, 2]) * 2.  # S=4*qx
                w = (rep[2, 1] - rep[1, 2]) / S
                workarr[0] = 0.25 * S
                workarr[1] = (rep[0, 1] + rep[1, 0]) / S
                workarr[2] = (rep[0, 2] + rep[2, 0]) / S
            elif rep[1, 1] > rep[2, 2]:
                S = np.sqrt(1. + rep[1, 1] - rep[0, 0] - rep[2, 2]) * 2.  # S=4*qy
                w = (rep[0, 2] - rep[2, 0]) / S
                workarr[0] = (rep[0, 1] + rep[1, 0]) / S
                workarr[1] = 0.25 * S
                workarr[2] = (rep[1, 2] + rep[2, 1]) / S
            else:
                S = np.sqrt(1. + rep[2, 2] - rep[0, 0] - rep[1, 1]) * 2.  # S=4*qz
                w = (rep[1, 0] - rep[0, 1]) / S
                workarr[0] = (rep[0, 2] + rep[2, 0]) / S
                workarr[1] = (rep[1, 2] + rep[2, 1]) / S
                workarr[2] = 0.25 * S
            if w < 0:
                workarr *= -1.
            if np.abs(w) >= 1. or not np.allclose(np.sqrt(np.sum(np.square(workarr)) + np.square(w)), 1.):
               print 'strange mat to vers result for:', rep
            assert np.allclose(np.sqrt(np.sum(np.square(workarr)) + np.square(w)), 1.)
            assert np.sum(np.square(workarr)) <= 1. or np.allclose(np.sum(np.square(workarr)), 1.)
    return retarr


cpdef versor_to_axis_angle_ipr(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 3 == 0
    cdef np.ndarray[np.float32_t, ndim=1] vec, recvec
    cdef np.float32_t comp_s, recrot, vecsqsum
    cdef int sample_idx, rot_idx
    for sample_idx in range(anglearr.shape[0]):
        for rot_idx in range(0, anglearr.shape[1], 3):
            vec = anglearr[sample_idx, rot_idx:rot_idx + 3]
            vecsqsum = np.sum(np.square(vec))
            assert vecsqsum <= 1. or np.allclose(vecsqsum, 1.)
            # Sidestep numerics.
            vecsqsum = np.clip(vecsqsum, 0., 1.)
            w = np.sqrt(1. - vecsqsum)
            #print 'w recovered:', w
            angle = 2. * np.arccos(w)
            #print 'angle recovered:', angle
            s = np.sqrt(1. - w ** 2)
            if s < 0.001:
                # if s is close to zero then the direction of the axis is not important.
                vec[:] = 1., 0., 0.
            else:
                vec[:] = vec / s * angle
            """
            comp_s = np.sqrt(1. - (np.sum(np.square(vec))))
            if comp_s < 0.001:
                # if s is close to zero then the direction of the axis is not important.
                vec[:] = 1., 0., 0.
            else:
                recrot = 2. * np.arctan2(np.linalg.norm(vec), comp_s)
                recvec = vec / np.sin(recrot / 2.)
                vec[:] = recvec / np.linalg.norm(recvec) * recrot
            """
    return anglearr


cpdef axis_angle_to_versor_ipr(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 3 == 0
    cdef np.ndarray[np.float32_t, ndim=1] vec, quat, rep
    cdef np.float32_t angle
    cdef int sample_idx, rot_idx
    quat = np.zeros((4,), dtype='float32')
    for sample_idx in range(anglearr.shape[0]):
        for rot_idx in range(0, anglearr.shape[1], 3):
            rep = anglearr[sample_idx, rot_idx:rot_idx + 3]
            angle = np.linalg.norm(rep)
            #print 'angle orig:', angle
            vec = rep / angle
            #print 'vec orig:', vec
            quat[0] = np.cos(angle / 2.)
            quat[1:] = vec * np.sin(angle / 2.)
            if quat[0] < 0.:
                quat = -quat
            rep[:] = quat[1:]
            #print 'w orig:', quat[0]
    return anglearr


cpdef axis_angle_to_euler_angle_ipr(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 3 == 0
    cdef int startpos
    cdef np.float32_t angle
    cdef np.ndarray[np.float32_t, ndim=1] rep
    cdef np.float32_t x, y, z, s, c, t
    cdef np.float32_t heading, attitude, bank
    for startpos in range(0, anglearr.size, 3):
        rep = anglearr.flat[startpos:startpos + 3]
        angle = np.linalg.norm(rep)
        x, y, z = rep / angle
        s = np.sin(angle)
        c = np.cos(angle)
        t = 1. - c
        if ((x*y*t + z*s) > 0.998):  # north pole singularity detected
            heading = 2. * np.arctan2(x * np.sin(angle / 2.),
                                      np.cos(angle / 2.))
            attitude = np.pi / 2.
            bank = 0.
        elif ((x*y*t + z*s) < -0.998):  # south pole singularity detected
            heading = -2. * np.arctan2(x * np.sin(angle / 2.),
                                       np.cos(angle / 2.))
            attitude = -np.pi / 2.
            bank = 0
        else:
            heading = np.arctan2(y * np.sin(angle) - x * z * (1. - np.cos(angle)),
                                 1. - (y**2 + z**2) * (1. - np.cos(angle)))
            attitude = np.arcsin(x * y * (1. - np.cos(angle)) + z * np.sin(angle))
            bank = np.arctan2(x * np.sin(angle)- y * z * (1. - np.cos(angle)),
                              1. - (x**2 + z**2) * (1. - np.cos(angle)))
        anglearr.flat[startpos:startpos + 3] = heading, attitude, bank
    return anglearr


cpdef euler_angle_to_axis_angle_ipr(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 3 == 0
    cdef int startpos
    cdef np.float32_t heading, attitude, bank
    cdef np.float32_t c1, c2, c3, s1, s2, s3, c1c2, s1s2, w
    cdef np.float32_t x, y, z, angle, norm
    for startpos in range(0, anglearr.size, 3):
        heading, attitude, bank = anglearr.flat[startpos:startpos + 3]
        c1 = np.cos(heading / 2.)
        s1 = np.sin(heading / 2.)
        c2 = np.cos(attitude / 2.)
        s2 = np.sin(attitude / 2.)
        c3 = np.cos(bank / 2.)
        s3 = np.sin(bank / 2.)
        c1c2 = c1 * c2
        s1s2 = s1 * s2
        w = c1c2 * c3 - s1s2 * s3
        x = c1c2 * s3 + s1s2 * c3
        y = s1 * c2 * c3 + c1 * s2 * s3
        z = c1 * s2 * c3 - s1 * c2 * s3
        angle = 2 * np.arccos(w)
        norm = x*x+y*y+z*z
        if (norm < 0.001):
            x = 1.
            y, z = 0., 0.
        else:
            norm = np.sqrt(norm)
            x /= norm
            y /= norm
            z /= norm
        anglearr.flat[startpos:startpos + 3] = x * angle, y * angle, z * angle
    return anglearr


cpdef abs_to_dist(np.ndarray[np.float32_t, ndim=2] absarr,
                  np.ndarray[np.int_t, ndim=2] combinations):
    assert combinations.shape[1] == 2
    cdef int sample_idx, feat_comb_idx
    cdef np.ndarray[np.float32_t, ndim=2] landmark_pos
    cdef np.ndarray[np.int_t, ndim=1] feat_comb
    cdef np.ndarray[np.float32_t, ndim=2] retarr
    retarr = np.zeros_like(absarr)
    for sample_idx in range(absarr.shape[0]):
        landmark_pos = absarr[sample_idx].reshape((2, 91))
        for feat_comb_idx, feat_comb in enumerate(combinations):
            retarr[sample_idx,
                   feat_comb_idx * 2:feat_comb_idx*2+2] = \
                landmark_pos[:, feat_comb[0]] - landmark_pos[:, feat_comb[1]]
    return retarr


class ProjectionError(Exception):
    pass


cpdef project_to_rot(np.ndarray[np.float32_t, ndim=1] anglearr):
    assert anglearr.shape[0] == 9
    cdef np.ndarray[np.float32_t, ndim=2] ret_arr    
    #cdef np.ndarray[np.float32_t, ndim=1] x, y, z
    cdef np.float32_t error
    ret_arr = polar(anglearr.reshape((3, 3)))[0]
    if np.linalg.det(ret_arr) < 0.:
       raise ProjectionError("Could not recover rotation, but only a reflection!")
    return ret_arr, np.linalg.norm(anglearr.reshape((3, 3)) - ret_arr)
    """
    x, y, z = anglearr.copy()
    # Determine the error.
    error = np.dot(x, y)
    # Resolve it.
    anglearr[0] = x - (error / 2.) * y
    anglearr[1] = y - (error / 2.) * x
    anglearr[2] = np.cross(anglearr[0], anglearr[1])
    # Normalize.
    #anglearr[0] /= np.linalg.norm(anglearr[0])
    #anglearr[1] /= np.linalg.norm(anglearr[1])
    #anglearr[2] /= np.linalg.norm(anglearr[2])
    #anglearr[0] = 0.5 * (3. - np.dot(anglearr[0], anglearr[0])) * anglearr[0]
    #anglearr[1] = 0.5 * (3. - np.dot(anglearr[1], anglearr[1])) * anglearr[1]
    #anglearr[2] = 0.5 * (3. - np.dot(anglearr[2], anglearr[2])) * anglearr[2]
    return anglearr, error
    """

cpdef project_to_rot_nofailure_ipr(np.ndarray[np.float32_t, ndim=2] anglearr):
    assert anglearr.shape[1] % 9 == 0
    cdef int startpos
    for startpos in range(0, anglearr.size, 9):
        anglearr.flat[startpos:startpos+9] = cv2.Rodrigues(
            cv2.Rodrigues(anglearr.flat[startpos:startpos+9].reshape((3, 3)))[0])[0].reshape((9,))
    return anglearr


cpdef versor_mean(versors):
    assert len(versors) > 0
    cdef np.ndarray[np.float32_t, ndim=1] res, firstvers, fullvers
    cdef np.ndarray[np.float32_t, ndim=2] workvers
    cpdef np.float32_t dnvers, w, w_res, vecsqsum
    dnvers = 1. / float(len(versors))

    vecsqsum = np.sum(np.square(versors[0]))
    assert vecsqsum <= 1. or np.allclose(vecsqsum, 1.)
    # Sidestep numerics.
    vecsqsum = np.clip(vecsqsum, 0., 1.)
    w = np.sqrt(1. - vecsqsum)

    firstvers = np.zeros((4,), dtype=np.float32)
    firstvers[0] = w
    firstvers[1:] = versors[0]
    res = np.atleast_2d(versors[0])[0] * dnvers
    w_res = w * dnvers
    fullvers = np.zeros((4,), dtype=np.float32)
    for workvers in versors:
        vecsqsum = np.sum(np.square(workvers))
        assert vecsqsum <= 1. or np.allclose(vecsqsum, 1.)
        # Sidestep numerics.
        vecsqsum = np.clip(vecsqsum, 0., 1.)
        w = np.sqrt(1. - vecsqsum)
        fullvers[0] = w
        fullvers[1:] = workvers
        if np.dot(fullvers, firstvers) < 0.:
            fullvers *= -1.
        res += np.atleast_2d(workvers)[0] * dnvers
        w_res += w * dnvers
    # Normalize.
    fullvers[0] = w_res
    fullvers[1:] = res
    fullvers /= np.linalg.norm(fullvers)
    if fullvers[0] < 0.:
        fullvers *= -1.
    return np.atleast_2d(fullvers[1:])
