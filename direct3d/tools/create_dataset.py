#!/usr/bin/env python2
"""Create 2D to 3D datasets from selected SMPL fits."""
# pylint: disable=invalid-name, wrong-import-order
import os
import os.path as path
import sys
import itertools
import logging
import csv
import cPickle as pickle

import numpy as np
import scipy
import scipy.io as sio
import cv2
import click
import opendr.camera as _odr_c
import tqdm
import h5py

from clustertools.log import LOGFORMAT
from up_tools.model import (robust_person_size, rlswap_landmarks_91,landmark_mesh_91,
                            connections_landmarks_91, dots_landmarks_91, get_crop)  # pylint: disable=no-name-in-module
from up_tools.mesh import Mesh
from up_tools.camera import (rotateY as rotateY,  # pylint: disable=unused-import
                             rotateX as rotateX)  # pylint: disable=unused-import
from up_tools.visualization import visualize_pose
sys.path.insert(0, path.join(path.abspath(path.dirname(__file__)),
                             '..', '..'))
from config import SMPL_FP, DIRECT3D_DATA_FP, UP3D_FP
try:
    # Robustify against setup.
    from smpl.serialization import load_model
except ImportError:
    # pylint: disable=import-error
    try:
        from psbody.smpl.serialization import load_model
    except ImportError:
        sys.path.insert(0, SMPL_FP)
        from smpl_webuser.serialization import load_model


LOGGER = logging.getLogger(__name__)
DSET_ROOT_FP = DIRECT3D_DATA_FP
MODEL_NEUTRAL_PATH = path.join(
    path.dirname(__file__), '..', '..', 'models', '3D',
    'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
MODEL_NEUTRAL = load_model(MODEL_NEUTRAL_PATH)
_TEMPLATE_MESH = Mesh(filename=path.join(
    path.dirname(__file__), '..', '..',
    'models', '3D',
    'template.ply'))


if not path.exists(DSET_ROOT_FP):
    os.makedirs(DSET_ROOT_FP)



def get_joints(indir):
    """Load the poses from an annotation tool dataset folder."""
    if path.exists(path.join(indir, 'joints.mat')):
        joints = sio.loadmat(path.join(indir, 'joints.mat'))['joints']
    else:
        joints = np.load(path.join(indir, 'joints.npz'))['poses']
        if 'mpii' in indir:
            LOGGER.info("Using mpii joint set.")
            joints = joints[:, [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9], :]
    if joints.shape[0] > 3:
        joints = joints.transpose((1, 0, 2))
    LOGGER.info("Joints for %d poses available.", joints.shape[2])
    return joints



def get_landmark_positions(stored_parameter_fp,  # pylint: disable=too-many-locals, too-many-arguments
                           resolution,
                           resolution_orig,  # pylint: disable=unused-argument
                           landmarks,
                           trans=(0, 0),  # pylint: disable=unused-argument
                           scale=1.,
                           steps_x=3, steps_y=12):
    """Get landmark positions for a given image."""
    with open(stored_parameter_fp, 'rb') as inf:
        stored_parameters = pickle.load(inf)
    orig_pose = np.array(stored_parameters['pose']).copy()
    orig_rt = np.array(stored_parameters['rt']).copy()
    orig_trans = np.array(stored_parameters['trans']).copy()
    orig_t = np.array(stored_parameters['t']).copy()
    model = MODEL_NEUTRAL
    model.betas[:len(stored_parameters['betas'])] = stored_parameters['betas']
    mesh = _TEMPLATE_MESH
    # Use always the image center for rendering.
    orig_t[0] = 0.
    orig_t[1] = 0.
    orig_t[2] /= scale
    # Prepare for rendering.
    angles_y = np.linspace(0., 2. * (1. - 1. / steps_y) * np.pi, steps_y)
    elevation_maxextent = (steps_x - 1) // 2 * 0.2 * np.pi
    angles_x = np.linspace(-elevation_maxextent,
                           elevation_maxextent,
                           steps_x)
    if steps_x == 1:
        # Assume plain layout.
        angles_x = (0.,)
    angles = itertools.product(angles_y, angles_x)
    landmark_positions = []
    full_parameters = []
    for angle_y, angle_x in angles:
        stored_parameters['rt'] = orig_rt.copy()
        stored_parameters['rt'][0] += angle_x
        stored_parameters['rt'][1] += angle_y
        #######################################################################
        # Zero out camera translation and rotation and move this information
        # to the body root joint rotations and 'trans' parameter.
        #print orig_pose[:3]
        cam_rdg, _ = cv2.Rodrigues(np.array(stored_parameters['rt']))
        per_rdg, _ = cv2.Rodrigues(np.array(orig_pose)[:3])
        resrot, _ = cv2.Rodrigues(np.dot(per_rdg, cam_rdg.T))
        restrans = np.dot(-np.array(orig_trans),
                          cam_rdg.T) + np.array(orig_t)
        stored_parameters['pose'][:3] = (-resrot).flat
        stored_parameters['trans'][:] = restrans
        stored_parameters['rt'][:] = [0, 0, 0]
        stored_parameters['t'][:] = [0, 0, 0]
        #######################################################################
        # Get the full rendered mesh.
        model.pose[:] = stored_parameters['pose']
        model.trans[:] = stored_parameters['trans']
        mesh.v = model.r
        mesh_points = mesh.v[tuple(landmarks.values()),]
        # Get the skeleton joints.
        J_onbetas = model.J_regressor.dot(mesh.v)
        skeleton_points = J_onbetas[(8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20),]
        # Do the projection.
        camera = _odr_c.ProjectPoints(
            rt=stored_parameters['rt'],
            t=stored_parameters['t'],
            f=(stored_parameters['f'], stored_parameters['f']),
            c=np.array(resolution) / 2.,
            k=np.zeros(5))
        camera.v = np.vstack((skeleton_points, mesh_points))
        full_parameters.append(list(stored_parameters['betas']) +
                               list(stored_parameters['pose']) +
                               list(stored_parameters['trans']) +
                               list(stored_parameters['rt']) +
                               list(stored_parameters['t']) +
                               [stored_parameters['f']])
        landmark_positions.append(list(camera.r.T.copy().flat))
    return landmark_positions, full_parameters


def add_dataset(dset_fp, list_ids, up3d_fp,  # pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-branches, unused-argument
                train_dset, val_dset, test_dset,
                train_spec, val_spec, test_spec,
                target_person_size, landmarks, train_crop, test_crop,
                train_steps_x, train_steps_y,
                running_idx,
                only_missing=False):
    """Add a dataset to the collection."""
    test_ids = [int(id_[1:6]) for id_ in test_spec]
    train_ids = [int(id_[1:6]) for id_ in train_spec]
    val_ids = [int(id_[1:6]) for id_ in val_spec]
    LOGGER.info("Split: %d train, %d val, %d test.",
                len(train_ids), len(val_ids), len(test_ids))
    LOGGER.info("Writing dataset...")
    for im_idx in tqdm.tqdm(train_ids + val_ids + test_ids):
        image = scipy.misc.imread(path.join(up3d_fp, '%05d_image.png' % (im_idx)))
        with open(path.join(up3d_fp, '%05d_fit_crop_info.txt' % (im_idx)), 'r') as inf:
            cropinfo = [int(val) for val in inf.readline().strip().split()]
            fac_y = cropinfo[0] / float(cropinfo[3] - cropinfo[2])
            fac_x = cropinfo[1] / float(cropinfo[5] - cropinfo[4])
            rec_scale = np.mean([fac_x, fac_y])
            rec_x = cropinfo[4]
            rec_y = cropinfo[2]
        assert image.ndim == 3
        out_exists = (path.exists(path.join(dset_fp, '%05d_image.png' % (running_idx))) and
                      path.exists(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx))))
        joints = np.load(path.join(up3d_fp, '%05d_joints.npy' % (im_idx)))
        joints = np.vstack((joints, np.all(joints > 0, axis=0)[None, :]))
        person_size = robust_person_size(joints)
        norm_factor = float(target_person_size) / person_size
        joints[:2, :] *= norm_factor
        image = scipy.misc.imresize(image, norm_factor, interp='bilinear')
        if im_idx in test_ids:
            crop = test_crop
        else:
            crop = train_crop
        if image.shape[0] > crop or image.shape[1] > crop:
            LOGGER.debug("Image (original %d, here %d) too large (%s)! Cropping...",
                         im_idx, running_idx, str(image.shape[:2]))
            person_center = np.mean(joints[:2, joints[2, :] == 1], axis=1)
            crop_y, crop_x = get_crop(image, person_center, crop)
            image = image[crop_y[0]:crop_y[1],
                          crop_x[0]:crop_x[1], :]
            assert image.shape[0] == crop or image.shape[1] == crop, (
                "Error cropping image (original %d, here %d)!" % (im_idx,
                                                                  running_idx))
        else:
            crop_x = [0, image.shape[1]]
            crop_y = [0, image.shape[0]]
        assert image.shape[0] <= crop and image.shape[1] <= crop and image.shape[2] == 3, (
            "Wrong image shape (original %d, here %d)!" % (im_idx, running_idx))
        if not (only_missing and out_exists):
            if im_idx in test_ids:
                steps_x = 1
                steps_y = 1
            else:
                steps_x = train_steps_x
                steps_y = train_steps_y
            LOGGER.debug('Crop infos: x: %s, y: %s', str(crop_x), str(crop_y))
            landmark_pos_list, full_parameter_list = get_landmark_positions(
                path.join(up3d_fp, '%05d_body.pkl' % (im_idx)),
                (image.shape[1],
                 image.shape[0]),
                (cropinfo[1],
                 cropinfo[0]),
                landmarks,
                trans=(-crop_x[0] - rec_x,  # pylint: disable=line-too-long
                       -crop_y[0] - rec_y),  # pylint: disable=line-too-long
                scale=norm_factor / rec_scale,  # pylint: disable=line-too-long
                steps_x=steps_x,
                steps_y=steps_y)
        if im_idx in train_ids:
            append_dset = train_dset
        elif im_idx in val_ids:
            append_dset = val_dset
        elif im_idx in test_ids:
            append_dset = test_dset
        for rend_idx, (landmark_pos, full_parameters) in enumerate(  # pylint: disable=unused-variable
                zip(landmark_pos_list, full_parameter_list)):
            append_dset.resize(append_dset.shape[0] + 1, axis=0)
            append_dset[-1, :] = landmark_pos + full_parameters
        running_idx += 1
    return running_idx


@click.command()
@click.argument("suffix", type=click.STRING)
@click.option("--target_person_size", type=click.INT, default=500)
@click.option("--train_steps_x", type=click.INT, default=3,
              help="Number of steps around x axis (elevation).")
@click.option("--train_steps_y", type=click.INT, default=12,
              help="Number of steps around y axis (azimuth).")
@click.option("--crop", type=click.INT, default=513,
              help="Crop size for the images.")
@click.option("--test_crop", type=click.INT, default=513,
              help="Crop size for the images.")
@click.option("--only_missing", type=click.BOOL, default=False, is_flag=True,
              help="Only rewrite missing images.")
@click.option("--up3d_fp", type=click.Path(file_okay=False, readable=True),
              default=UP3D_FP,
              help="Path to the UP3D folder that you want to use.")
def cli(suffix, target_person_size=500, train_steps_x=3, train_steps_y=12,  # pylint: disable=too-many-locals, too-many-arguments
        crop=513, test_crop=513, only_missing=False, up3d_fp=UP3D_FP):
    """Create 2D to 3D dataset from select SMPL fits."""
    np.random.seed(1)
    if test_crop < target_person_size or crop < target_person_size:
        LOGGER.critical("Too small crop size!")
        raise Exception("Too small crop size!")
    LOGGER.info("Loading landmark mapping...")
    landmark_mapping = landmark_mesh_91
    n_landmarks = len(landmark_mapping) + 12
    LOGGER.info("Creating 2D to 3D dataset with %d landmarks with target "
                "person size %f and suffix `%s`.",
                n_landmarks, target_person_size, suffix)
    assert ' ' not in suffix
    dset_fromroot = path.join(str(n_landmarks), str(target_person_size), suffix)
    dset_fp = path.join(DSET_ROOT_FP, dset_fromroot)
    if path.exists(dset_fp):
        if not click.confirm("Dataset folder exists: `%s`! Continue?" % (dset_fp)):
            return
    else:
        os.makedirs(dset_fp)
    LOGGER.info("Creating HDF5 data files...")
    n_features = 91 * 2
    n_targets = 92
    train_out = h5py.File(path.join(dset_fp, "train.hdf5"), "w")
    train_dset = train_out.create_dataset("2dto3d",
                                          (0, n_features + n_targets),
                                          dtype='float32',
                                          maxshape=(None, n_features + n_targets))
    val_out = h5py.File(path.join(dset_fp, "val.hdf5"), "w")
    val_dset = val_out.create_dataset("2dto3d",
                                      (0, n_features + n_targets),
                                      dtype='float32',
                                      maxshape=(None, n_features + n_targets))
    test_out = h5py.File(path.join(dset_fp, "test.hdf5"), "w")
    test_dset = test_out.create_dataset("2dto3d",
                                        (0, n_features + n_targets),
                                        dtype='float32',
                                        maxshape=(None, n_features + n_targets))
    list_ids = np.zeros((4,), dtype='int')
    with open(path.join(up3d_fp, 'train.txt'), 'r') as f:
        train_spec = [line.strip() for line in f.readlines()]
    with open(path.join(up3d_fp, 'val.txt'), 'r') as f:
        val_spec = [line.strip() for line in f.readlines()]
    with open(path.join(up3d_fp, 'test.txt'), 'r') as f:
        test_spec = [line.strip() for line in f.readlines()]

    running_idx = 0
    LOGGER.info("Processing...")
    #dset_dir = path.join(FIT_TOOL_FP, 'datasets', 'lsp')
    #image_dir = path.join(dset_dir, 'images')
    running_idx = add_dataset(
        dset_fp,
        list_ids,
        up3d_fp,
        train_dset,
        val_dset,
        test_dset,
        train_spec,
        val_spec,
        test_spec,
        target_person_size, landmark_mapping,
        crop, test_crop, train_steps_x, train_steps_y, running_idx,
        only_missing=only_missing)
    LOGGER.info("Done.")
    train_out.close()
    val_out.close()
    test_out.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("opendr.lighting").setLevel(logging.WARN)
    cli()  # pylint: disable=no-value-for-parameter

