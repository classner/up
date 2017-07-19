import numpy as np
from mesh import Mesh

__all__ = ['Sphere']


class Sphere:
    def __init__(self, center, radius):
        if(center.flatten().shape != (3,)):
            raise Exception("Center should have size(1,3) instead of %s" % center.shape)
        self.center = center.flatten()
        self.radius = radius

    def __str__(self):
        return "%s:%s" % (self.center, self.radius)

    def to_mesh(self, color=np.array([1., 0., 0.])):
        v = np.array([[0.0000, -1.000, 0.0000], [0.7236, -0.447, 0.5257],
                      [-0.278, -0.447, 0.8506], [-0.894, -0.447, 0.0000],
                      [-0.278, -0.447, -0.850], [0.7236, -0.447, -0.525],
                      [0.2765, 0.4472, 0.8506], [-0.723, 0.4472, 0.5257],
                      [-0.720, 0.4472, -0.525], [0.2763, 0.4472, -0.850],
                      [0.8945, 0.4472, 0.0000], [0.0000, 1.0000, 0.0000],
                      [-0.165, -0.850, 0.4999], [0.4253, -0.850, 0.3090],
                      [0.2629, -0.525, 0.8090], [0.4253, -0.850, -0.309],
                      [0.8508, -0.525, 0.0000], [-0.525, -0.850, 0.0000],
                      [-0.688, -0.525, 0.4999], [-0.162, -0.850, -0.499],
                      [-0.688, -0.525, -0.499], [0.2628, -0.525, -0.809],
                      [0.9518, 0.0000, -0.309], [0.9510, 0.0000, 0.3090],
                      [0.5876, 0.0000, 0.8090], [0.0000, 0.0000, 1.0000],
                      [-0.588, 0.0000, 0.8090], [-0.951, 0.0000, 0.3090],
                      [-0.955, 0.0000, -0.309], [-0.587, 0.0000, -0.809],
                      [0.0000, 0.0000, -1.000], [0.5877, 0.0000, -0.809],
                      [0.6889, 0.5257, 0.4999], [-0.262, 0.5257, 0.8090],
                      [-0.854, 0.5257, 0.0000], [-0.262, 0.5257, -0.809],
                      [0.6889, 0.5257, -0.499], [0.5257, 0.8506, 0.0000],
                      [0.1626, 0.8506, 0.4999], [-0.425, 0.8506, 0.3090],
                      [-0.422, 0.8506, -0.309], [0.1624, 0.8506, -0.499]]);

        f = np.array([[15, 3, 13], [13, 14, 15], [2, 15, 14], [13, 1, 14], [17, 2, 14], [14, 16, 17],
                      [6, 17, 16], [14, 1, 16], [19, 4, 18], [18, 13, 19], [3, 19, 13], [18, 1, 13],
                      [21, 5, 20], [20, 18, 21], [4, 21, 18], [20, 1, 18], [22, 6, 16], [16, 20, 22],
                      [5, 22, 20], [16, 1, 20], [24, 2, 17], [17, 23, 24], [11, 24, 23], [23, 17, 6],
                      [26, 3, 15], [15, 25, 26], [7, 26, 25], [25, 15, 2], [28, 4, 19], [19, 27, 28],
                      [8, 28, 27], [27, 19, 3], [30, 5, 21], [21, 29, 30], [9, 30, 29], [29, 21, 4],
                      [32, 6, 22], [22, 31, 32], [10, 32, 31], [31, 22, 5], [33, 7, 25], [25, 24, 33],
                      [11, 33, 24], [24, 25, 2], [34, 8, 27], [27, 26, 34], [7, 34, 26], [26, 27, 3],
                      [35, 9, 29], [29, 28, 35], [8, 35, 28], [28, 29, 4], [36, 10, 31], [31, 30, 36],
                      [9, 36, 30], [30, 31, 5], [37, 11, 23], [23, 32, 37], [10, 37, 32], [32, 23, 6],
                      [39, 7, 33], [33, 38, 39], [12, 39, 38], [38, 33, 11], [40, 8, 34], [34, 39, 40],
                      [12, 40, 39], [39, 34, 7], [41, 9, 35], [35, 40, 41], [12, 41, 40], [40, 35, 8],
                      [42, 10, 36], [36, 41, 42], [12, 42, 41], [41, 36, 9], [38, 11, 37], [37, 42, 38],
                      [12, 38, 42], [42, 37, 10]]) - 1

        return Mesh(v=v * self.radius + self.center, f=f, vc=np.tile(color, (v.shape[0], 1)))

    def has_inside(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

    def intersects(self, sphere):
        return np.linalg.norm(sphere.center - self.center) < (self.radius + sphere.radius)

    def intersection_vol(self, sphere):
        if not self.intersects(sphere):
            return 0
        d = np.linalg.norm(sphere.center - self.center)
        R, r = (self.radius, sphere.radius) if (self.radius > sphere.radius) else (sphere.radius, self.radius)
        if R >= (d + r):
            return (4 * np.pi * (r ** 3)) / 3

        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return (np.pi * (R + r - d) ** 2 * (d ** 2 + 2 * d * r - 3 * r * r + 2 * d * R + 6 * r * R - 3 * R * R)) / (12 * d)

from chained import *
#from body.scape.scapefunctions import relative_rotations_to_scape_rotations


def rodrigues2rotmat(r):
    R = np.zeros((3, 3))
    r_skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    theta = np.linalg.norm(r)
    return np.identity(3) + np.sin(theta) * r_skew + (1 - np.cos(theta)) * r_skew.dot(r_skew)


class SphereModel(Chained):

    def __init__(self, mesh, kin_tree, max_spheres_ratio=50,
                 samples_per_part=50000, spheres_per_part=10):

        self.kin_tree = kin_tree
        pnames = kin_tree.get_part_names()
        mesh = Mesh(filename=hand_path)
        if not hasattr(mesh, 'segm'):
            raise Exception('Sphere models require a segmentation of the triangles')
        part_arr_spheres, frontiers_avg = pack(mesh.v, [mesh.f[mesh.segm[p]] for p in pnames],
                                               max_spheres_ratio, samples_per_part, spheres_per_part)

        karr = kin_tree.get_kin_arr()

        part_origin = np.zeros((len(pnames), 3))
        for par_child in karr[:, 1:].T:
            if tuple(par_child) in frontiers_avg:
                part_origin[par_child[1], :] = frontiers_avg[tuple(par_child)]
            else:
                raise Exception('Neighboring parts in kintree should have common vertices')

        # root part doesn't have a frontier defining the origin; pick the mean vertex
        part_origin[0] = np.mean(mesh.v[kin_tree.vert_idxs([pnames[0]], mesh)], axis=0)

        # construct resting transformation matrices for every part
        self.spheres = []
        self.centers = []
        for i_part in range(len(part_arr_spheres)):
            part_spheres = []
            for sphere in part_arr_spheres[i_part]:
                part_spheres.append(Sphere(sphere[0:3], sphere[3]))
            self.spheres.append(part_spheres)
            self.centers.append(np.array(part_arr_spheres[i_part])[:, :3].T)

        self.kin_tree.set_part_origins(pnames, part_origin)

    def apply_pose(self, pose, scale):

        pnames = self.kin_tree.get_part_names()
        if pose.shape != (len(self.spheres), 3):
            raise Exception('Pose should have shape %s' % ((len(self.spheres) * 3, 1)))

        Rrels = [rodrigues2rotmat(r) for r in pose]  # assuming pose (Nx3)
        self.kin_tree.set_Rrels(pnames, Rrels)

        self.kin_tree.computeTxs()

        new_centers = self.kin_tree.applyTxs(pnames, self.centers)
        new_spheres = []
        for i_part, centers_part in enumerate(new_centers):
            part_spheres = []
            for i_center, center in enumerate(centers_part.T):
                part_spheres.append(Sphere(center, self.spheres[i_part][i_center].radius))
            new_spheres.append(part_spheres)

        return new_spheres
