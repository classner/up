#!/usr/bin/env python2
"""Run the fitting of the body model using discriminative models."""
# pylint: disable=invalid-name, wrong-import-order
import os as _os
import sys as _sys
import os.path as _path
from copy import copy as _copy
import glob as _glob
import logging as _logging
import cPickle as _pickle
import joblib
import multiprocessing as _multiprocessing
import pymp as _pymp
from time import time as _time

import cv2 as _cv2
import numpy as _np
import scipy as _scipy
import click as _click
import opendr.camera as _odr_c
from opendr.camera import ProjectPoints as _ProjectPoints, ProjectPoints3D as _ProjectPoints3D
import opendr.lighting as _odr_l
from opendr.renderer import ColoredRenderer as _ColoredRenderer
from up_tools.mesh import Mesh
from up_tools.camera import rotateY as _rotateY
_sys.path.insert(0, _path.join(_path.dirname(__file__), '..'))
from config import SMPL_FP
_sys.path.insert(0, SMPL_FP)
try:
    # Robustify against setup.
    from smpl.serialization import load_model
    from smpl.lbs import global_rigid_transformation as _global_rigid_transformation
except ImportError:
    # pylint: disable=import-error
    try:
        from psbody.smpl.serialization import load_model
        from psbody.smpl.lbs import global_rigid_transformation as _global_rigid_transformation
    except:
        from smpl_webuser.serialization import load_model
        from smpl_webuser.lbs import global_rigid_transformation as _global_rigid_transformation

from up_tools.model import reduction_91tolsp, robust_person_size, landmark_mesh_91, get_pose_names
import pyximport; pyximport.install()  # pylint: disable=multiple-statements
from conversions import project_to_rot_nofailure_ipr, matrix_to_versor, versor_to_axis_angle_ipr  # pylint: disable=import-error
from clustertools.log import LOGFORMAT
import fasteners
from fit_forest import lmset_to_use, create_featseltuple
import tqdm
import chumpy as _ch


_MODEL_NEUTRAL_PATH = _path.join(
    _path.dirname(__file__), '..',
    'models', '3D',
    'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
_MODEL_NEUTRAL = load_model(_MODEL_NEUTRAL_PATH)
_TEMPLATE_MESH = Mesh(filename=_path.join(
    _path.dirname(__file__), '..',
    'models', '3D',
    'template.ply'))
_LANDMARK_MAPPING = landmark_mesh_91
# Guess the focal length.
_FLENGTH_GUESS = 5000.
_MOD_PATH = _os.path.abspath(_os.path.dirname(__file__))
# Models:
_MODEL_SHAPE = None
_MODEL_ROT = None
_DTA_ROT = None
_MODEL_POSE = None
_MODEL_T = None
# Estimators:
_DEPTH_EST = None
_ROT_EST = None
_SHAPE_EST = None
_POSE_EST = None

_DEBUG = False
if _DEBUG:
    _DEBUG_SFX = "_debug"
else:
    _DEBUG_SFX = "_final"
_LOGGER = _logging.getLogger(__name__)


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
    rn.frustum = {'near':near, 'far':far,
                  'height':h, 'width':w}
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
        light_pos=_rotateY(_np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=_np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += _odr_l.LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(_np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=_np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += _odr_l.LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(_np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=_np.array([.7, .7, .7]))
    return rn.r


def _depth_estimator(landmark_array, ret_array, command_queue, command_lock, ret_queue):
    global _MODEL_T  # pylint: disable=global-statement
    _LOGGER.info("Depth estimator starting up...")
    if _MODEL_T is None:
        _LOGGER.info("Loading depth estimation model...")
        _MODEL_T = joblib.load(
            _path.join(_path.dirname(__file__), '..', 'models', '2dto3d',
                       'separate_regressors', 'forest_82%s.z'%(_DEBUG_SFX)))
    ret_queue.put('ready')
    _LOGGER.info("Depth estimator ready.")
    rel_ids = create_featseltuple(lmset_to_use[(82, 85)])
    while True:
        cmd = command_queue.get()
        if cmd == 'shutdown':
            break
        with command_lock:
            landmarks = landmark_array.reshape((1, 182))[0:1, rel_ids].copy()
        ret_array[...] = _MODEL_T.predict(landmarks)
        ret_queue.put('done')
    _LOGGER.info("Depth estimator shut down.")


def normalize_versor(anglevec):
    """Normalize a vector of 3D angles to versors."""
    assert len(anglevec) % 3 == 0
    for startpos in range(0, len(anglevec), 3):
        rep = anglevec[startpos:startpos + 3]
        angle = _np.linalg.norm(rep)
        vec = rep / angle
        quat = _np.array([_np.cos(angle / 2.)] + list(vec * _np.sin(angle / 2.)))
        if quat[0] < 0.:
            quat = -quat
        anglevec[startpos:startpos + 3] = quat[1:]


def _rot_estimator(landmark_array, ret_array, command_queue, command_lock, ret_queue):
    global _MODEL_ROT, _DTA_ROT  # pylint: disable=global-statement, global-variable-not-assigned
    _LOGGER.info("Rotation estimator starting up...")
    if _MODEL_ROT is None:
        _LOGGER.info("Loading rotation estimation model...")
        _MODEL_ROT = joblib.load(
            _path.join(_path.dirname(__file__), '..', 'models', '2dto3d',
                       'separate_regressors', 'forest_10%s.z'%(_DEBUG_SFX)))
    ret_queue.put('ready')
    _LOGGER.info("Rotation estimator ready.")
    rel_ids = create_featseltuple(lmset_to_use[(10, 13)])
    while True:
        cmd = command_queue.get()
        if cmd == 'shutdown':
            break
        with command_lock:
            landmarks = landmark_array.reshape((1, 182))[0:1, rel_ids].copy()
        #qidx = _MODEL_ROT.query(landmarks.reshape((1, 182)), return_distance=False)[0, 0]
        #ret_array[...] = _DTA_ROT[qidx, 10:13]
        ret_array[...] = versor_to_axis_angle_ipr(
            matrix_to_versor(
                project_to_rot_nofailure_ipr(
                    _MODEL_ROT.predict(landmarks).astype('float32'))))
        ret_queue.put('done')
    _LOGGER.info("Rotation estimator shut down.")


def _shape_estimator(landmark_array, ret_array, command_queue, command_lock, ret_queue):
    global _MODEL_SHAPE  # pylint: disable=global-statement
    _LOGGER.info("Shape estimator starting up...")
    if _MODEL_SHAPE is None:
        _LOGGER.info("Loading shape estimation model...")
        _MODEL_SHAPE = joblib.load(
            _path.join(_path.dirname(__file__), '..', 'models', '2dto3d',
                       'separate_regressors', 'forest_0%s.z'%(_DEBUG_SFX)))
    ret_queue.put('ready')
    _LOGGER.info("Shape estimator ready.")
    rel_ids = create_featseltuple(lmset_to_use[(0, 10)])
    while True:
        cmd = command_queue.get()
        if cmd == 'shutdown':
            break
        with command_lock:
            landmarks = landmark_array.reshape((1, 182))[0:1, rel_ids].copy()
        ret_array[...] = _MODEL_SHAPE.predict(landmarks)
        ret_queue.put('done')
    _LOGGER.info("Shape estimator shut down.")


def _pose_estimator(landmark_array, ret_array, command_queue, command_lock, ret_queue):
    global _MODEL_POSE  # pylint: disable=global-statement
    _LOGGER.info("Pose estimator starting up...")
    if _MODEL_POSE is None:
        _LOGGER.info("Loading pose estimation model...")
        _MODEL_POSE = {}
        for start_idx in tqdm.tqdm(range(13, 79, 3)):
            _MODEL_POSE[start_idx] = joblib.load(
                _path.join(_path.dirname(__file__), '..', 'models', '2dto3d',
                           'separate_regressors',
                           'forest_%d%s.z' % (start_idx, _DEBUG_SFX)))
    ret_queue.put('ready')
    _LOGGER.info("Pose estimator ready.")
    rel_ids = {}
    for start_idx in range(13, 79, 3):
        rel_ids[start_idx] = create_featseltuple(lmset_to_use[(start_idx, start_idx+3)])
    while True:
        cmd = command_queue.get()
        if cmd == 'shutdown':
            break
        with command_lock:
            landmarks = landmark_array.reshape((1, 182)).copy()
        for start_idx in range(13, 79, 3):
            ret_array[start_idx - 13:start_idx - 10] = versor_to_axis_angle_ipr(
                matrix_to_versor(
                    project_to_rot_nofailure_ipr(
                        _MODEL_POSE[start_idx].predict(
                            landmarks[0:1, rel_ids[start_idx]]).astype('float32'))))
        ret_queue.put('done')
    _LOGGER.info("Pose estimator shut down.")


def get_landmark_positions(stored_parameters,  # pylint: disable=too-many-locals, too-many-arguments
                           resolution,
                           landmarks):
    """Get landmark positions for a given image."""
    model = _MODEL_NEUTRAL
    model.betas[:len(stored_parameters['betas'])] = stored_parameters['betas']
    mesh = _TEMPLATE_MESH
    # Get the full rendered mesh.
    model.pose[:] = stored_parameters['pose']
    model.trans[:] = stored_parameters['trans']
    mesh.v = model.r
    mesh_points = mesh.v[tuple(landmarks.values()),]
    # Get the skeleton joints.
    J_onbetas = model.J_regressor.dot(mesh.v)
    skeleton_points = J_onbetas[(8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20),]
    camera = _odr_c.ProjectPoints(
        rt=stored_parameters['rt'],
        t=stored_parameters['t'],
        f=(stored_parameters['f'], stored_parameters['f']),
        c=_np.array(resolution) / 2.,
        k=_np.zeros(5))
    camera.v = _np.vstack((skeleton_points, mesh_points))
    landmark_positions = camera.r.T.copy()
    return landmark_positions


def _versor_to_axis_angle(valarr):
    """Convert an array with stacked versors to an axis-angle representation."""
    assert len(valarr) % 3 == 0
    for vecstart in range(0, len(valarr), 3):
        vec = valarr[vecstart:vecstart + 3]
        comp_s = _np.sqrt(1.- (_np.sum(_np.square(vec))))
        recrot = 2. * _np.arctan2(_np.linalg.norm(vec), comp_s)
        recvec = vec / _np.sin(recrot / 2.)
        vec[:] = recvec / _np.linalg.norm(recvec) * recrot
    return valarr


def _fit_rot_trans(model,  # pylint: disable=too-many-arguments, too-many-locals
                   j2d,
                   center,
                   init_t,
                   init_pose,
                   conf,
                   flength):
    """Find a rotation and translation to minimize the projection error of the pose."""
    opt_pose = _ch.array(init_pose)  # pylint: disable=no-member
    opt_trans = _ch.array(init_t)  # pylint: disable=no-member
    (_, A_global) = _global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=_ch)
    Jtr = _ch.vstack([g[:3, 3] for g in A_global]) + opt_trans  # pylint: disable=no-member

    cam = _ProjectPoints(f=_np.array([flength, flength]),
                         rt=_np.zeros(3),
                         t=_np.zeros(3),
                         k=_np.zeros(5),
                         c=center)
    cam_3d = _ProjectPoints3D(f=_np.array([flength, flength]),
                              rt=_np.zeros(3),
                              t=_np.zeros(3),
                              k=_np.zeros(5),
                              c=center)
    cam.v = Jtr
    cam_3d.v = Jtr
    free_variables = [opt_pose[:3], opt_trans]  # Optimize global rotation and translation.
    j2d_ids = range(12)
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
    #_CID_TORSO = [2, 3, 8, 9]
    #torso_smpl_ids = [2, 1, 17, 16]
    _ch.minimize(
        {
            'j2d': (j2d.T[j2d_ids] - cam[smpl_ids]).T * conf[j2d_ids],
            # 'dev^2': 1e2 * (opt_trans[2] - init_t[2])
        },
        x0=free_variables,
        method='dogleg',
        callback=None, #on_step_vis,
        options={'maxiter': 100, 'e_3': .0001, 'disp': 0})
    _LOGGER.debug("Global rotation: %s, global translation: %s.",
                  str(opt_pose[:3].r), str(opt_trans.r))
    _LOGGER.debug("Points 3D: %s.",
                  str(cam_3d.r[:10, :]))
    _LOGGER.debug("Points 2D: %s.",
                  str(cam.r[:10, :]))
    return opt_pose[:3].r, opt_trans.r, cam_3d.r[:, 2].min(), cam_3d.r[:, 2].max()


def run_single_fit(img,  # pylint: disable=too-many-statements, too-many-locals
                   j2d,
                   scale,
                   do_degrees=None):
    """Run the fit for one specific image."""
    global _DEPTH_EST, _SHAPE_EST, _ROT_EST, _POSE_EST  # pylint: disable=global-statement
    assert j2d.shape[0] == 3
    assert j2d.shape[1] == 91
    conf = j2d[2, :].copy().reshape((-1,))
    j2d = j2d[:2, :].copy()
    j2d_norm = j2d * scale
    # Center the data.
    mean = _np.mean(j2d_norm, axis=1)
    j2d_norm = (j2d_norm.T - mean + 513. / 2.).T
    _LOGGER.debug("Running fit...")
    if do_degrees is None:
        do_degrees = []
    # Prepare the estimators if necessary.
    if _DEPTH_EST is None:
        _DEPTH_EST = [None,
                      _pymp.shared.array(j2d.shape, dtype='float32'),
                      _pymp.shared.array((3,), dtype='float32'),
                      _pymp.shared.queue(),
                      _pymp.shared.lock(),
                      _pymp.shared.queue()]
        _DEPTH_EST[0] = _multiprocessing.Process(target=_depth_estimator,
                                                 args=tuple(_DEPTH_EST[1:]))
        _DEPTH_EST[0].start()
        _DEPTH_EST[5].get()
    if _ROT_EST is None:
        _ROT_EST = [None,
                    _pymp.shared.array(j2d.shape, dtype='float32'),
                    _pymp.shared.array((3,), dtype='float32'),
                    _pymp.shared.queue(),
                    _pymp.shared.lock(),
                    _pymp.shared.queue()]
        _ROT_EST[0] = _multiprocessing.Process(target=_rot_estimator,
                                               args=tuple(_ROT_EST[1:]))
        _ROT_EST[0].start()
        _ROT_EST[5].get()
    if _SHAPE_EST is None:
        _SHAPE_EST = [None,
                      _pymp.shared.array(j2d.shape, dtype='float32'),
                      _pymp.shared.array((10,), dtype='float32'),
                      _pymp.shared.queue(),
                      _pymp.shared.lock(),
                      _pymp.shared.queue()]
        _SHAPE_EST[0] = _multiprocessing.Process(target=_shape_estimator,
                                                 args=tuple(_SHAPE_EST[1:]))
        _SHAPE_EST[0].start()
        _SHAPE_EST[5].get()
    if _POSE_EST is None:
        _POSE_EST = [None,
                     _pymp.shared.array(j2d.shape, dtype='float32'),
                     _pymp.shared.array((69,), dtype='float32'),
                     _pymp.shared.queue(),
                     _pymp.shared.lock(),
                     _pymp.shared.queue()]
        _POSE_EST[0] = _multiprocessing.Process(target=_pose_estimator,
                                                args=tuple(_POSE_EST[1:]))
        _POSE_EST[0].start()
        _POSE_EST[5].get()
    # Copy the data to the processes.
    with _POSE_EST[4]:
        _POSE_EST[1][...] = j2d_norm
    with _SHAPE_EST[4]:
        _SHAPE_EST[1][...] = j2d_norm
    with _ROT_EST[4]:
        _ROT_EST[1][...] = j2d_norm
    with _DEPTH_EST[4]:
        _DEPTH_EST[1][...] = j2d_norm
    # Run it.
    before_fit = _time()
    _POSE_EST[3].put('go')
    _ROT_EST[3].put('go')
    _SHAPE_EST[3].put('go')
    _DEPTH_EST[3].put('go')
    _LOGGER.info("Running...")
    _DEPTH_EST[5].get()
    _POSE_EST[5].get()
    _SHAPE_EST[5].get()
    _ROT_EST[5].get()
    _LOGGER.info("Prediction available in %ss.", str(_time() - before_fit))
    # Extract the results.
    pose = _np.zeros((72,), dtype='float32')
    betas = _np.zeros((10,), dtype='float32')
    trans = _np.zeros((3,), dtype='float32')
    with _POSE_EST[4]:
        pose[3:] = _POSE_EST[2]
    with _SHAPE_EST[4]:
        betas[:] = _SHAPE_EST[2]
    with _ROT_EST[4]:
        pose[:3] = _ROT_EST[2]
    with _DEPTH_EST[4]:
        trans[:] = _DEPTH_EST[2]
    trans[2] *= scale
    # Get the projected landmark locations from the model.
    param_dict = {'t': [0, 0, 0],
                  'rt': [0, 0, 0],
                  'f': _FLENGTH_GUESS,
                  'pose': pose,
                  'trans': trans,
                  'betas': betas}
    # Optimize depth and global rotation.
    opt_globrot, opt_trans, dmin, dmax = _fit_rot_trans(_MODEL_NEUTRAL,
                                                        j2d,
                                                        [img.shape[1] // 2,
                                                         img.shape[0] // 2],
                                                        trans,
                                                        pose,
                                                        conf,
                                                        _FLENGTH_GUESS)
    pose[:3] = opt_globrot
    trans[:] = opt_trans
    """
    proj_landmark_positions = get_landmark_positions(param_dict,
                                                     (513, 513),
                                                     _LANDMARK_MAPPING)
    # Get the right offset to match the original.
    offset = _np.mean(j2d, axis=1) - _np.mean(proj_landmark_positions, axis=1)
    """
    # Render the optimized mesh.
    _LOGGER.info("Rendering...")
    mesh = _copy(_TEMPLATE_MESH)
    model = _MODEL_NEUTRAL
    model.betas[:len(betas)] = betas
    # Get the full rendered mesh.
    model.pose[:] = pose
    model.trans[:] = trans
    mesh.v = model.r
    mesh.vc = [.7, .7, .9]
    base_mesh_v = mesh.v.copy()
    images = []
    for deg in do_degrees:
        mesh.v = _rotateY(base_mesh_v.copy(), deg)
        rn = create_renderer(w=img.shape[1],
                             h=img.shape[0],
                             near=dmin - 1.,
                             far=dmax + 1.,
                             rt=[0., 0., 0.],
                             t=[0., 0., 0.],
                             f=[_FLENGTH_GUESS, _FLENGTH_GUESS],
                             c=[img.shape[1] // 2,
                                img.shape[0] // 2])  # + offset[1]])
        light_yrot = _np.radians(120)
        im = (simple_renderer(rn=rn,
                              meshes=[mesh],
                              yrot=light_yrot) * 255.).astype('uint8')
        images.append(im)
    #param_dict['j2d'] = (proj_landmark_positions.T + offset).T
    _LOGGER.info("Estimation done.")
    return param_dict, images


@_click.command()
@_click.argument('image_name', type=_click.Path(exists=True, readable=True))
@_click.option('--out_name',
               type=_click.Path(exists=False, writable=True),
               default=None,
               help='The name of the output archive to generate.')
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help='The ending to use for the images to read, if a folder is specified.',
               default='.png')
@_click.option('--only_missing', type=_click.BOOL, default=False, is_flag=True,
               help='Only run the fit for images with no fit results.')
@_click.option('--allow_subsampling', type=_click.BOOL, default=False, is_flag=True,
               help='Dont raise for images without pose.')
def cli(image_name, out_name=None,  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        folder_image_suffix='.png',
        only_missing=False,
        allow_subsampling=False):
    """Get a 3D body model fit."""
    if _os.path.isdir(image_name):
        processing_folder = True
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = sorted(_glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix)))
        images = [im for im in images if not im.endswith('vis.png')]
        pose_names = get_pose_names(images)
    else:
        processing_folder = False
        images = [image_name]
        pose_names = get_pose_names(images)
    for image_name, pose_name in zip(images, pose_names):
        if not _path.exists(pose_name) and allow_subsampling:
            continue
        if out_name is None or processing_folder:
            out_name = image_name + '_body_directseparate.pkl'
        vis_name = out_name + '_vis.png'
        work_name = out_name + '_working'
        lock_name = _path.join(_path.dirname(work_name), 'work_lock')
        if only_missing and _path.exists(out_name) and _path.exists(vis_name):
            continue
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
        core_pose = pose[:3, reduction_91tolsp].copy()
        core_pose[2, :] = core_pose[2, :] >= 0.  # no threshold right now
        person_size_est = robust_person_size(core_pose)
        size_factor = 500. / person_size_est
        _LOGGER.info("Predicting the 3D body on `%s` (saving to `%s`).",
                     image_name, out_name)
        image = _scipy.misc.imread(image_name)
        if image.ndim == 2:
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, 3::-1]
        params, vis = run_single_fit(image,  # pylint: disable=unused-variable
                                     pose[:3, :],
                                     size_factor,
                                     do_degrees=[0.])
        with open(out_name, 'w') as outf:
            _pickle.dump(params, outf)
        _cv2.imwrite(vis_name, vis[0])
        _os.remove(work_name)
    _LOGGER.info("Shutting down...")
    if _POSE_EST is not None:
        _POSE_EST[3].put('shutdown')
        _POSE_EST[0].join()
    if _SHAPE_EST is not None:
        _SHAPE_EST[3].put('shutdown')
        _SHAPE_EST[0].join()
    if _DEPTH_EST is not None:
        _DEPTH_EST[3].put('shutdown')
        _DEPTH_EST[0].join()
    if _ROT_EST is not None:
        _ROT_EST[3].put('shutdown')
        _ROT_EST[0].join()

if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO, format=LOGFORMAT)
    _logging.getLogger("opendr.lighting").setLevel(_logging.WARN)
    _logging.getLogger("OpenGL.GL.shaders").setLevel(_logging.WARN)
    if _DEBUG:
        _LOGGER.critical("DEBUG mode enabled!")
    cli()  # pylint: disable=no-value-for-parameter
