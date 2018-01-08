#!/usr/bin/env python2
"""Run the fitting of the body model."""
# pylint: disable=invalid-name
import os as _os
import os.path as _path
import socket
import sys as _sys
import glob as _glob
import logging as _logging
import cPickle as _pickle
from copy import copy as _copy  # pylint: disable=import-error
from time import time as _time
import fasteners
from scipy.sparse.linalg import cg
import cv2 as _cv2
try:
    # OpenCV 2
    import cv as _cv
    _DIST_L1 = _cv.CV_DIST_L1  # pylint: disable=no-member
    _DIST_MASK = _cv.CV_DIST_MASK_PRECISE  # pylint: disable=no-member
except:  # pylint: disable=bare-except
    # OpenCV 3
    _DIST_L1 = _cv2.DIST_L1
    _DIST_MASK = _cv2.DIST_MASK_PRECISE
import numpy as _np
import chumpy as _ch
import scipy as _scipy
import click as _click

from opendr.camera import ProjectPoints as _ProjectPoints
import opendr.lighting as _odr_l
from opendr.renderer import ColoredRenderer as _ColoredRenderer

_sys.path.insert(0, _path.join(_path.dirname(__file__), '..'))
from config import SMPL_FP
_sys.path.insert(0, SMPL_FP)
from up_tools.mesh import Mesh as _Mesh
from up_tools.robustifiers import GMOf as _GMOf
from up_tools.model import joints_lsp, connections_lsp, get_pose_names
try:
    # Robustify against setup.
    from smpl.serialization import load_model as _load_model
    from smpl.lbs import global_rigid_transformation as _global_rigid_transformation
    from smpl.verts import verts_decorated
except ImportError:
    # pylint: disable=import-error
    try:
        from psbody.smpl.serialization import load_model as _load_model
        from psbody.smpl.lbs import global_rigid_transformation as _global_rigid_transformation
        from psbody.smpl.verts import verts_decorated
    except:
        from smpl_webuser.serialization import load_model as _load_model
        from smpl_webuser.lbs import global_rigid_transformation as _global_rigid_transformation
        from smpl_webuser.verts import verts_decorated
from up_tools.sphere_collisions import SphereCollisions as _SphereCollisions
from up_tools.max_mixture_prior import MaxMixtureCompletePrior as _MaxMixtureCompletePrior
from up_tools.model import landmarks_91, landmark_mesh_91
from up_tools.camera import rotateY

from clustertools.log import LOGFORMAT

_LOGGER = _logging.getLogger(__name__)

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       fake

# guess the focal length
_FLENGTH_GUESS = 5000. #2500.
# initial value for the camera translation (in meters)
_T_GUESS = _np.array([0, 0, 20.])
# number of model shape coefficients to optimize
_N_BETAS = 10
_MOD_PATH = _os.path.abspath(_os.path.dirname(__file__))
# Models:
_MODEL_NEUTRAL_FNAME = _os.path.join(
    _MOD_PATH, '..', 'models', '3D',
    'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
_MODEL_NEUTRAL = _load_model(_MODEL_NEUTRAL_FNAME)
# Torso joint IDs (used for estimating camera position).
_CID_TORSO = [landmarks_91.rhip,
              landmarks_91.lhip,
              landmarks_91.rshoulder,
              landmarks_91.lshoulder]
# Head position correction.
_HEAD_CORR_FNAME = _os.path.join(
    _MOD_PATH, '..', 'models', '3D', 'head_vid_heva.pkl')
with open(_HEAD_CORR_FNAME, 'r') as f:
    _HEAD_REGR = _pickle.load(f)
_REGRESSORS_FNAME = _os.path.join(
    _MOD_PATH, '..', 'models', '3D', 'regressors_locked_normalized_hybrid.npz')
_REGRESSORS = _np.load(_REGRESSORS_FNAME)
# Project the regressors on the first 10 beta dimensions.
_REGP = dict()
_REGP['v2lens'] = _REGRESSORS['v2lens']
_REGP['betas2lens'] = _REGRESSORS['betas2lens'][tuple(range(10) + [
    300,
]), :]
_REGP['v2rads'] = _REGRESSORS['v2rads']
_REGP['betas2rads'] = _REGRESSORS['betas2rads'][tuple(range(10) + [
    300,
]), :]
_REGRESSORS = _REGP



def create_renderer(w=640,  # pylint: disable=too-many-arguments
                    h=480,
                    rt=_np.zeros(3),
                    t=_np.zeros(3),
                    f=None,  # pylint: disable=redefined-outer-name
                    c=None,
                    k=None,
                    near=.5,
                    far=10.):
    """Create a colored renderer."""
    f = _np.array([w, w])/2. if f is None else f
    c = _np.array([w, h])/2. if c is None else c
    k = _np.zeros(5)         if k is None else k
    rn = _ColoredRenderer()
    rn.camera = _ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near':near, 'far':far, 'height':h, 'width':w}
    return rn

def simple_renderer(rn, meshes, yrot=0):
    """Create a renderer, optionally with texture."""
    mesh = meshes[0]
    if hasattr(rn, 'texture_image'):
        if not hasattr(mesh, 'ft'):
            mesh.ft = _copy(mesh.f)
            vt = _copy(mesh.v[:, :2])
            vt -= _np.min(vt, axis=0).reshape((1, -1))
            vt /= _np.max(vt, axis=0).reshape((1, -1))
            mesh.vt = vt
        mesh.texture_filepath = rn.texture_image
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc,
               ft=mesh.ft, vt=mesh.vt, bgcolor=_np.ones(3))
    else:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, bgcolor=_np.ones(3))

    for next_mesh in meshes[1:]:
        _stack_with(rn, next_mesh)  # pylint: disable=undefined-variable

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = _odr_l.LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(_np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=_np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += _odr_l.LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(_np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=_np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += _odr_l.LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(_np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=_np.array([.7, .7, .7]))
    return rn.r

# --------------------Camera estimation --------------------
def guess_init(model,  # pylint: disable=too-many-locals, too-many-arguments
               focal_length,
               j2d,
               weights,
               init_pose,
               use_gt_guess):
    """Initialize the camera depth using the torso points."""
    cids = _np.arange(0, 12)
    j2d_here = j2d[cids]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
    # The initial pose is a t-pose with all body parts planar to the camera
    # (i.e., they have their maximum length).
    opt_pose = _ch.array(init_pose)  # pylint: disable=no-member
    (_, A_global) = _global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=_ch)
    Jtr = _ch.vstack([g[:3, 3] for g in A_global])  # pylint: disable=no-member
    Jtr = Jtr[smpl_ids].r

    if use_gt_guess:
        # More robust estimate:
        # Since there is no fore'lengthening', only foreshortening, take the longest
        # visible body part as a robust estimate in the case of noise-free data.
        longest_part = None
        longest_part_length = -1.
        for conn in connections_lsp:
            if (conn[0] < 12 and conn[1] < 12 and
                    weights[conn[0]] > 0. and weights[conn[1]] > 0.):
                part_length_proj_gt = _np.sqrt(_np.sum(
                    _np.array(j2d_here[conn[0]] - j2d_here[conn[1]]) ** 2, axis=0))
                if part_length_proj_gt > longest_part_length:
                    longest_part_length = part_length_proj_gt
                    longest_part = conn
        #length2d = _np.sqrt(_np.sum(
        #    _np.array(j2d_here[longest_part[0]] - j2d_here[longest_part[1]])**2, axis=0))
        part_length_3D = _np.sqrt(_np.sum(
            _np.array(Jtr[longest_part[0]] - Jtr[longest_part[1]]) ** 2, axis=0))
        est_d = focal_length * (part_length_3D / longest_part_length)
    else:
        diff3d = _np.array([Jtr[9] - Jtr[3], Jtr[8] - Jtr[2]])
        mean_height3d = _np.mean(_np.sqrt(_np.sum(diff3d**2, axis=1)))
        diff2d = _np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
        mean_height2d = _np.mean(_np.sqrt(_np.sum(diff2d**2, axis=1)))
        f = focal_length  # pylint: disable=redefined-outer-name
        if mean_height2d == 0.:
            _LOGGER.warn("Depth can not be correctly estimated. Guessing wildly.")
            est_d = 60.
        else:
            est_d = f * (mean_height3d / mean_height2d)
    init_t = _np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,  # pylint: disable=too-many-arguments, too-many-locals
                      j2d,
                      center,
                      img,
                      init_t,
                      init_pose,
                      conf,
                      is_gt,
                      flength=1000.,
                      pix_thsh=25.,
                      viz=False):
    """Initialize the camera."""
    # try to optimize camera translation and rotation based on torso joints
    # right shoulder, left shoulder, right hip, left hip
    torso_cids = _CID_TORSO
    torso_smpl_ids = [2, 1, 17, 16]

    # initialize camera rotation and translation
    rt = _ch.zeros(3)  # pylint: disable=no-member
    _LOGGER.info('Initializing camera: guessing translation via similarity')
    init_t = guess_init(model, flength, j2d, conf, init_pose, is_gt)
    t = _ch.array(init_t)  # pylint: disable=no-member

    # check how close the shoulders are
    try_both_orient = _np.linalg.norm(j2d[8] - j2d[9]) < pix_thsh

    opt_pose = _ch.array(init_pose)  # pylint: disable=no-member
    (_, A_global) = _global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=_ch)
    Jtr = _ch.vstack([g[:3, 3] for g in A_global])  # pylint: disable=no-member

    # initialize the camera
    cam = _ProjectPoints(f=_np.array([flength, flength]),
                         rt=rt, t=t, k=_np.zeros(5), c=center)

    # we are going to project the SMPL joints
    cam.v = Jtr

    if viz:
        viz_img = img.copy()

        # draw the target joints
        for coord in _np.around(j2d).astype(int):
            if (coord[0] < img.shape[1] and coord[0] >= 0 and
                    coord[1] < img.shape[0] and coord[1] >= 0):
                _cv2.circle(viz_img, tuple(coord), 3, [0, 255, 0])

        import matplotlib.pyplot as plt
        plt.ion()
        # draw optimized joints at each iteration
        def on_step(_):
            """Draw a visualization."""
            plt.figure(1, figsize=(5, 5))
            plt.subplot(1, 1, 1)
            viz_img = img.copy()
            for coord in _np.around(cam.r[torso_smpl_ids]).astype(int):
                if (coord[0] < viz_img.shape[1] and coord[0] >= 0 and
                        coord[1] < viz_img.shape[0] and coord[1] >= 0):
                    _cv2.circle(viz_img, tuple(coord), 3, [0, 0, 255])
            plt.imshow(viz_img[:, :, ::-1])
            plt.draw()
            plt.show()
    else:
        on_step = None
    free_variables = [cam.t, opt_pose[:3]]  # pylint: disable=no-member
    _ch.minimize([(j2d[torso_cids] - cam[torso_smpl_ids]).T * conf[torso_cids],
                  1e2*(cam.t[2] - init_t[2])],  # pylint: disable=no-member
                 # The same for verbose output.
                 #{'cam': (j2d[torso_cids] - cam[torso_smpl_ids]).T * conf[torso_cids],
                 # # Reduce the weight here to avoid the 'small people' problem.
                 # 'cam_t': 1e2*(cam.t[2]-init_t[2])},  # pylint: disable=no-member
                 x0=free_variables,
                 method='dogleg',
                 callback=on_step,
                 options={'maxiter': 100, 'e_3': .0001, 'disp': 0})
    if viz:
        plt.ioff()
    return (cam, try_both_orient, opt_pose[:3].r)

# --------------------Core optimization --------------------
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       prior,
                       try_both_orient,
                       body_orient,
                       exp_logistic,
                       n_betas=10,
                       inner_penetration=False,
                       silh=None,
                       conf=None,
                       viz=False):
    """Run the optimization."""
    if silh is not None:
        raise NotImplementedError("Silhouette fitting is not supported in "
                                  "this code release due to dependencies on "
                                  "proprietary code for the "
                                  "distance computation.")
    t0 = _time()
    # define the mapping LSP joints -> SMPL joints
    if j2d.shape[0] == 14:
	cids = range(12) + [13]
    elif j2d.shape[0] == 91:
        cids = range(j2d.shape[0])
    else:
        raise Exception("Unknown number of joints: %d! Mapping not defined!" % j2d.shape[0])
    # joint ids for SMPL
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
    # weight given to each joint during optimization;
    if j2d.shape[0] == 14:
        weights = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    else:
        weights = [1] * (len(smpl_ids) + len(landmark_mesh_91))
    # The non-skeleton vertex ids are added later.

    if try_both_orient:
        flipped_orient = _cv2.Rodrigues(body_orient)[0].dot(
            _cv2.Rodrigues(_np.array([0., _np.pi, 0]))[0])
        flipped_orient = _cv2.Rodrigues(flipped_orient)[0].ravel()
        orientations = [body_orient, flipped_orient]
    else:
        orientations = [body_orient]

    if try_both_orient:
        errors = []
    svs = []
    cams = []
    # rends = []
    for o_id, orient in enumerate(orientations):
        # initialize betas
        betas = _ch.zeros(n_betas)  # pylint: disable=no-member

        init_pose = _np.hstack((orient, prior.weights.dot(prior.means)))

        # 2D joint error term
        # make the SMPL joint depend on betas
        Jdirs = _np.dstack([model.J_regressor.dot(
            model.shapedirs[:, :, i]) for i in range(len(betas))])
        # pylint: disable=no-member
        J_onbetas = _ch.array(Jdirs).dot(betas) + model.J_regressor.dot(model.v_template.r)

        # instantiate the model
        sv = verts_decorated(
            trans=_ch.zeros(3),
            pose=_ch.array(init_pose),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs)

        # get joint positions as a function of model pose, betas and trans
        (_, A_global) = _global_rigid_transformation(
            sv.pose, J_onbetas, model.kintree_table, xp=_ch)
        Jtr = _ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

        if j2d.shape[0] == 14:
            # add the "fake" joint for the head
            head_id = _HEAD_REGR[0]
            Jtr = _ch.vstack((Jtr, sv[head_id]))
            if o_id == 0:
                smpl_ids.append(len(Jtr)-1)
        else:
            # add the plain vertex IDs on the mesh surface.
            for vertex_id in landmark_mesh_91.values():
                Jtr = _ch.vstack((Jtr, sv[vertex_id]))
                # add the joint id
                # for SMPL it's the last one added
                if o_id == 0:
                    smpl_ids.append(len(Jtr)-1)
        weights = _np.array(weights, dtype=_np.float64)
        if conf is not None:
            weights *= conf[cids]

        # we'll project the joints on the image plane
        cam.v = Jtr

        # data term: difference between observed and estimated joints
        obj_j2d = lambda w, sigma: (w * weights.reshape((-1, 1)) *
                                    _GMOf((j2d[cids] - cam[smpl_ids]), sigma))
        # pose prior
        pprior = lambda w: w*prior(sv.pose)  # pylint: disable=cell-var-from-loop
        # joint angles prior
        # 55: left elbow, should bend -np.pi/2
        # 58: right elbow, should bend np.pi/2
        # 12: left knee, should bend np.pi/2
        # 15: right knee, should bend np.pi/2
        if exp_logistic:
            _LOGGER.info('USING LOGISTIC')
            # Skinny Logistic function. as 50-> inf we get a step function at
            # 0.1. (0.1) is a margin bc 0 is still ok.
            my_exp = lambda x: 1 / (1 + _ch.exp(100 * (0.1 +- x)))
        else:
            x_0 = 0 #10
            alpha = 10
            my_exp = lambda x: alpha * _ch.exp((x-x_0))  # pylint: disable=cell-var-from-loop


        obj_angle = lambda w: w*_ch.concatenate([my_exp(sv.pose[55]),  # pylint: disable=cell-var-from-loop
                                                 my_exp(-sv.pose[58]),  # pylint: disable=cell-var-from-loop
                                                 my_exp(-sv.pose[12]),  # pylint: disable=cell-var-from-loop
                                                 my_exp(-sv.pose[15])])  # pylint: disable=cell-var-from-loop

        if viz:
            from body.mesh.sphere import Sphere
            from body.mesh.meshviewer import MeshViewer
            import matplotlib.pyplot as plt

            # set up visualization
            # openGL window
            mv = MeshViewer(window_width=120, window_height=120)

            # and ids
            show_ids = _np.array(smpl_ids)[weights > 0]
            vc = _np.ones((len(Jtr), 3))
            vc[show_ids] = [0, 1, 0]

            plt.ion()

            def on_step(_):
                """Create visualization."""
                # show optimized joints in 3D
                # pylint: disable=cell-var-from-loop
                mv.set_dynamic_meshes([_Mesh(v=sv.r, f=[]),
                                       Sphere(center=cam.t.r,
                                              radius=.1).to_mesh()] \
                        + [Sphere(center=jc, radius=.01).to_mesh(vc[ijc])
                           for ijc, jc in enumerate(Jtr.r)])
                plt.figure(1, figsize=(10, 10))
                plt.subplot(1, 2, 1)
                # show optimized joints in 2D
                tmp_img = img.copy()
                for coord, target_coord in zip(_np.around(cam.r[smpl_ids]).astype(int),
                                               _np.around(j2d[cids]).astype(int)):
                    if (coord[0] < tmp_img.shape[1] and coord[0] >= 0 and
                            coord[1] < tmp_img.shape[0] and coord[1] >= 0):
                        _cv2.circle(tmp_img, tuple(coord), 3, [0, 0, 255])
                    if (target_coord[0] < tmp_img.shape[1] and target_coord[0] >= 0
                            and target_coord[1] < tmp_img.shape[0] and target_coord[1] >= 0):
                        _cv2.circle(tmp_img, tuple(target_coord), 3, [0, 255, 0])
                plt.imshow(tmp_img)
                plt.draw()
                plt.show()
            on_step(_)
        else:
            on_step = None

        sp = _SphereCollisions(pose=sv.pose,
                               betas=sv.betas,
                               model=model,
                               regs=_REGRESSORS)
        sp.no_hands = True
        # configuration used with conf joints
        opt_weights = zip([4.04*1e2, 4.04*1e2, 57.4, 4.78], [1e2, 5*1e1, 1e1, .5*1e1])

        for stage, (w, wbetas) in enumerate(opt_weights):
            _LOGGER.info('stage %01d', stage)
            objs = {}
            #if stage < 2:
            objs['j2d'] = obj_j2d(1., 100)  # TODO: evaluate.

            objs['pose'] = pprior(w)

            # WEIGHT FOR ANGLE
            if exp_logistic:
                # Set to high weight always.
                objs['pose_exp'] = obj_angle(5*1e3)
            else:
                objs['pose_exp'] = obj_angle(0.317*w)

            objs['betas'] = wbetas*betas
            if inner_penetration:
                objs['sph_coll'] = 1e3*sp
            try:
                _ch.minimize(objs.values(),
                             x0=[sv.betas, sv.pose],
                             method='dogleg',
                             callback=on_step,
                             options={
                                 'maxiter': 100,
                                 'e_3': .0001,
                                 'disp': 0})
            except AssertionError:
                # Divergence detected.
                _LOGGER.warn("Diverging optimization! Breaking!")
                break
        t1 = _time()
        _LOGGER.info('elapsed %.05f', (t1-t0))
        if try_both_orient:
            errors.append((objs['j2d'].r**2).sum())
        svs.append(sv)
        cams.append(cam)
        # rends.append(rend)
    if try_both_orient and errors[0] > errors[1]:
        choose_id = 1
    else:
        choose_id = 0
    if viz:
        plt.ioff()
    return (svs[choose_id],
            cams[choose_id].r,
            cams[choose_id].t.r,
            cams[choose_id].rt.r)


# pylint: disable=too-many-statements, unused-argument
def run_single_fit(img,
                   j2d,
                   conf,
                   inner_penetration=False,
                   silh=None,
                   scale_factor=1,
                   gender='neutral',
                   exp_logistic=False,
                   viz=False,
                   do_degrees=None,
                   is_gt_data=False):
    """Run the fit for one specific image."""
    model = _MODEL_NEUTRAL
    if silh is not None:
        if silh.ndim == 3:
            silh = _cv2.split(silh)[0]
        silh = _np.uint8(silh > 0)

    if do_degrees is None:
        do_degrees = []

    # create the pose prior (GMM over CMU)
    prior = _MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = _np.hstack((_np.zeros(3), prior.weights.dot(prior.means)))

    if scale_factor != 1:
        img = _cv2.resize(img, (img.shape[1] * scale_factor,
                                img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor

    # get the center of the image (needed to estimate camera parms)
    center = _np.array([img.shape[1]/2, img.shape[0]/2])

    # estimate the camera parameters
    (cam, try_both_orient, body_orient) = initialize_camera(
        model, j2d, center, img,
        _T_GUESS, init_pose, conf, is_gt_data, flength=_FLENGTH_GUESS, viz=viz)

    # fit
    (sv, opt_j2d, t, rt) = optimize_on_joints(  # pylint: disable=unused-variable
        j2d, model, cam, img, prior, try_both_orient,
        body_orient, exp_logistic, n_betas=_N_BETAS, conf=conf, viz=viz,
        inner_penetration=inner_penetration, silh=silh)

    # get the optimized mesh
    m = _Mesh(v=sv.r, f=model.f)
    m.vc = [.7, .7, .9]

    dist = _np.abs(cam.t.r[2] - _np.mean(sv.r, axis=0)[2])  # pylint: disable=no-member
    h = img.shape[0]
    w = img.shape[1]
    rn = create_renderer(w=w,
                         h=h,
                         near=1.,
                         far=20.+dist,
                         rt=cam.rt,  # pylint: disable=no-member
                         t=cam.t,  # pylint: disable=no-member
                         f=cam.f,  # pylint: disable=no-member
                         c=cam.c)  # pylint: disable=no-member
    light_yrot = _np.radians(120)
    images = []
    orig_v = sv.r
    for deg in do_degrees:
        aroundy = _cv2.Rodrigues(_np.array([0, _np.radians(deg), 0]))[0]
        center = orig_v.mean(axis=0)
        new_v = _np.dot((orig_v - center), aroundy)
        m.v = new_v + center
        # Now render.
        im = (simple_renderer(rn=rn,
                              meshes=[m],
                              yrot=light_yrot) * 255.).astype('uint8')
        images.append(im)
    # save to disk
    result = {'j2d': opt_j2d,
              't': t,
              'rt': rt,
              'f': _FLENGTH_GUESS,
              'pose': sv.pose.r,
              'betas': sv.betas.r,
              'trans': sv.trans.r}
    return sv, result, images


@_click.command()
@_click.argument('image_name', type=_click.Path(exists=True, readable=True))
@_click.option('--out_name',
               type=_click.Path(exists=False, writable=True),
               default=None,
               help='The name of the output archive to generate.')
@_click.option('--use_inner_penetration',
               is_flag=True,
               default=False,
               type=_click.BOOL,
               help='Use inner penetration analysis. Slows down computation.')
@_click.option('--use_silhouette',
               is_flag=True,
               default=False,
               type=_click.BOOL,
               help='Use silhouette for the fits. Not available for now.')
@_click.option('--is_gt',
               is_flag=True,
               default=False,
               type=_click.BOOL,
               help=('Assume joints and silhouettes to be ground truth. Use a '
                     'different, more robust method for camera depth estimation '
                     'that is more vulnerable to joint noise. Hence, be careful '
                     'to use low confidence (or 0 confidence) for bad joints!'))
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help='The ending to use for the images to read, if a folder is specified.',
               default='.png')
@_click.option('--only_missing',
               type=_click.BOOL, default=False, is_flag=True,
               help="Only run fits for images with missing results.")
@_click.option('--allow_subsampling',
               type=_click.BOOL, default=False, is_flag=True,
               help="Skip images without poses without complaints.")
def cli(image_name,
        out_name=None,
        use_inner_penetration=False,
        use_silhouette=False,
        is_gt=False,
        folder_image_suffix='.png',
        only_missing=False,
        allow_subsampling=False):
    """Get a 3D body model fit."""
    _LOGGER.info("Running on host `%s`.", socket.gethostname())
    if _os.path.isdir(image_name):
        processing_folder = True
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = sorted(_glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix)))
        images = [im for im in images if not im.endswith('vis.png')]
        pose_names = get_pose_names(images)
        rel_names = [image_name + '_reliability.npz' for image_name in images]
        #images = [im + '.npy.vis.png' for im in images]
    else:
        processing_folder = False
        images = [image_name]
        pose_names = get_pose_names(images)
        rel_names = [image_name + '_reliability.npz']
    for image_name, pose_name, agreement_name in zip(images, pose_names, rel_names):
        if allow_subsampling and not _os.path.exists(pose_name):
            continue
        if out_name is None or processing_folder:
            out_name = image_name + '_body.pkl'
        vis_name = out_name + '_vis.png'
        if only_missing and _os.path.exists(out_name) and _os.path.exists(vis_name):
            continue
        # Interprocess comm.
        work_name = out_name + '_working'
        lock_name = _path.join(_path.dirname(work_name), 'work_lock')
        with fasteners.InterProcessLock(lock_name):
            if _path.exists(work_name):
                continue
            else:
                with open(work_name, 'w') as outf:
                    outf.write('t')
        if pose_name.endswith('.npz'):
            pose = _np.load(pose_name)['pose']
        else:
            pose = _np.load(pose_name)
        try:
            agreement = _np.load(agreement_name)['reliability']
        except:  # pylint: disable=bare-except
            _LOGGER.warn("No agreement info found! Using all joints with their prob.!")
            #agreement = _np.ones((pose.shape[1],))
            agreement = pose[2, :]  # >= 0.8
        _LOGGER.info("Predicting the 3D body on `%s` (saving to `%s`).",
                     image_name, out_name)
        if use_inner_penetration:
            _LOGGER.info("Using inner penetration avoidance.")
        if use_silhouette:
            silh_npz_fname = image_name + '_segmentation.npz'
            _LOGGER.info("Using silhouette for fitting (%s).", silh_npz_fname)
            silh = _np.load(silh_npz_fname)['segmentation']
        else:
            silh = None
        image = _scipy.misc.imread(image_name)
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, 3::-1]
        body, params, vis = run_single_fit(image,  # pylint: disable=unused-variable
                                           pose[:2, :].T,
                                           agreement,
                                           inner_penetration=use_inner_penetration,
                                           silh=silh,
                                           do_degrees=[0.],
                                           is_gt_data=is_gt)
        with open(out_name, 'w') as outf:
            _pickle.dump(params, outf)
        _cv2.imwrite(vis_name.encode("ascii"), vis[0])
        _os.remove(work_name)


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO, format=LOGFORMAT)
    _logging.getLogger("opendr.lighting").setLevel(_logging.WARN)
    cli()  # pylint: disable=no-value-for-parameter
