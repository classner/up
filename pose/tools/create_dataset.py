#!/usr/bin/env python2
"""Create segmentation datasets from select SMPL fits."""
# pylint: disable=invalid-name
import os
import os.path as path
import sys
import logging
import csv
from collections import OrderedDict
from copy import copy as _copy
import cPickle as pickle

import numpy as np
import scipy
import scipy.io as sio
import click
import opendr.camera as _odr_c
import tqdm

from clustertools.log import LOGFORMAT

from up_tools.model import (robust_person_size, rlswap_lsp, connections_lsp,
                            rlswap_landmarks_91, get_crop, landmark_mesh_91)
import up_tools.visualization as vs
from up_tools.render_segmented_views import render_body_impl  # pylint: disable=unused-import
from up_tools.mesh import Mesh
try:
    # Robustify against setup.
    from smpl.serialization import load_model
except ImportError:
    # pylint: disable=import-error
    try:
        from psbody.smpl.serialization import load_model
    except ImportError:
        from smpl_webuser.serialization import load_model
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))
from config import POSE_DATA_FP, UP3D_FP


LOGGER = logging.getLogger(__name__)
DSET_ROOT_FP = POSE_DATA_FP
MODEL_NEUTRAL_PATH = path.join(
    path.dirname(__file__), '..', '..', 'models', '3D',
    'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
MODEL_NEUTRAL = load_model(MODEL_NEUTRAL_PATH)
_TEMPLATE_MESH = Mesh(filename=path.join(
    path.dirname(__file__), '..', '..',
    'models', '3D',
    'template.ply'))

if not path.exists(DSET_ROOT_FP):
    os.mkdir(DSET_ROOT_FP)



def get_landmark_positions(stored_parameter_fp, resolution, landmarks):
    """Get landmark positions for a given image."""
    with open(stored_parameter_fp, 'rb') as inf:
        stored_parameters = pickle.load(inf)
    # Pose the model.
    model = MODEL_NEUTRAL
    model.betas[:len(stored_parameters['betas'])] = stored_parameters['betas']
    model.pose[:] = stored_parameters['pose']
    model.trans[:] = stored_parameters['trans']
    mesh = _copy(_TEMPLATE_MESH)
    mesh.v = model.r
    mesh_points = mesh.v[tuple(landmarks.values()),]
    J_onbetas = model.J_regressor.dot(model.r)
    skeleton_points = J_onbetas[(8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20),]
    # Do the projection.
    camera = _odr_c.ProjectPoints(
        rt=stored_parameters['rt'],
        t=stored_parameters['t'],
        f=(stored_parameters['f'], stored_parameters['f']),
        c=np.array(resolution) / 2.,
        k=np.zeros(5))
    camera.v = np.vstack((skeleton_points, mesh_points))
    return camera.r.T.copy()


def add_dataset(dset_fp, dset_fromroot, list_ids, up3d_fp,  # pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-branches
                train_list_f, val_list_f, train_val_list_f, test_list_f, scale_f,
                train_spec, val_spec, test_spec,
                target_person_size, landmarks, train_crop, test_crop, running_idx,
                only_missing=False, with_rlswap=True, write_gtjoints_as_lm=False,
                human_annotations=False):
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
        assert image.ndim == 3
        out_exists = (path.exists(path.join(dset_fp, '%05d_image.png' % (running_idx))) and
                      path.exists(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx))))
        if with_rlswap and im_idx not in test_ids:
            out_exists = out_exists and (
                path.exists(path.join(dset_fp, '%05d_image.png' % (running_idx + 1))) and
                path.exists(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx + 1))))
        if not (only_missing and out_exists or write_gtjoints_as_lm):
            if human_annotations:
                landmark_pos = np.load(path.join(up3d_fp, '%05d_joints.npy' % (im_idx)))
            else:
                landmark_pos = get_landmark_positions(path.join(up3d_fp, '%05d_body.pkl' % (im_idx)),
                                                      (cropinfo[1], cropinfo[0]),
                                                      landmarks)
                fac_y = cropinfo[0] / float(cropinfo[3] - cropinfo[2])
                fac_x = cropinfo[1] / float(cropinfo[5] - cropinfo[4])
                landmark_pos[:2, :] /= np.mean([fac_x, fac_y])
                landmark_pos[0, :] += cropinfo[4]
                landmark_pos[1, :] += cropinfo[2]
        joints = np.load(path.join(up3d_fp, '%05d_joints.npy' % (im_idx)))
        joints = np.vstack((joints, np.all(joints > 0, axis=0)[None, :]))
        person_size = robust_person_size(joints)
        norm_factor = float(target_person_size) / person_size
        joints[:2, :] *= norm_factor
        if not (only_missing and out_exists or write_gtjoints_as_lm):
            landmark_pos[:2, :] *= norm_factor
        if write_gtjoints_as_lm:
            landmark_pos = joints.copy()
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
            landmark_pos[0, :] -= crop_x[0]
            landmark_pos[1, :] -= crop_y[0]
            assert image.shape[0] == crop or image.shape[1] == crop, (
                "Error cropping image (original %d, here %d)!" % (im_idx,
                                                                  running_idx))
        assert image.shape[0] <= crop and image.shape[1] <= crop and image.shape[2] == 3, (
            "Wrong image shape (original %d, here %d)!" % (im_idx, running_idx))
        vis_im = vs.visualize_pose(image, landmark_pos, scale=1.)
        if not (only_missing and out_exists):
            scipy.misc.imsave(path.join(dset_fp, '%05d_image.png' % (running_idx)), image)
            scipy.misc.imsave(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx)), vis_im)
        if with_rlswap and im_idx not in test_ids:
            if landmark_pos.shape[1] == 14:
                landmark_pos_swapped = landmark_pos[:, rlswap_lsp]
            else:
                landmark_pos_swapped = landmark_pos[:, rlswap_landmarks_91]
            landmark_pos_swapped[0, :] = image.shape[1] - landmark_pos_swapped[0, :]
            image_swapped = image[:, ::-1, :]
            # Use core visualization for 14 joints.
            vis_im_swapped = vs.visualize_pose(image_swapped,
                                               landmark_pos_swapped,
                                               scale=1)
            if not (only_missing and out_exists):
                scipy.misc.imsave(path.join(dset_fp, '%05d_image.png' % (running_idx + 1)),
                                  image_swapped)
                scipy.misc.imsave(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx + 1)),
                                  vis_im_swapped)
        list_fs = []
        list_id_ids = []
        if im_idx in train_ids:
            list_fs.append(train_val_list_f)
            list_id_ids.append(2)
            list_fs.append(train_list_f)
            list_id_ids.append(0)
        elif im_idx in val_ids:
            list_fs.append(train_val_list_f)
            list_id_ids.append(2)
            list_fs.append(val_list_f)
            list_id_ids.append(1)
        elif im_idx in test_ids:
            list_fs.append(test_list_f)
            list_id_ids.append(3)
        for list_f, list_id_idx in zip(list_fs, list_id_ids):
            # pylint: disable=bad-continuation
            list_f.write(
"""# %d
%s
3
%d
%d
%d
""" % (
    list_ids[list_id_idx],
    path.join('/' + dset_fromroot, '%05d_image.png' % (running_idx)),
    image.shape[0],
    image.shape[1],
    landmark_pos.shape[1]))
            for landmark_idx, landmark_point in enumerate(landmark_pos.T):
                list_f.write("%d %d %d\n" % (landmark_idx + 1,
                                             int(landmark_point[0]),
                                             int(landmark_point[1])))
            list_f.flush()
            list_ids[list_id_idx] += 1
        scale_f.write("%05d_image.png %f\n" % (running_idx, norm_factor))
        scale_f.flush()
        running_idx += 1
        if with_rlswap and im_idx not in test_ids:
            for list_f, list_id_idx in zip(list_fs, list_id_ids):
                # pylint: disable=bad-continuation
                list_f.write(
"""# %d
%s
3
%d
%d
%d
""" % (
    list_ids[list_id_idx],
    path.join('/' + dset_fromroot, '%05d_image.png' % (running_idx)),
    image.shape[0],
    image.shape[1],
    landmark_pos.shape[1]))
                for landmark_idx, landmark_point in enumerate(landmark_pos_swapped.T):
                    list_f.write("%d %d %d\n" % (landmark_idx + 1,
                                                 int(landmark_point[0]),
                                                 int(landmark_point[1])))
                list_f.flush()
                list_ids[list_id_idx] += 1
            scale_f.write("%05d_image.png %f\n" % (running_idx, norm_factor))
            scale_f.flush()
            running_idx += 1
    return running_idx


@click.command()
@click.argument("suffix", type=click.STRING)
@click.argument("target_person_size", type=click.INT)
@click.option("--crop", type=click.INT, default=513,  # Used to be 513.
              help="Crop size for the images.")
@click.option("--test_crop", type=click.INT, default=513,  # Used to be 513.
              help="Crop size for the images.")
@click.option("--only_missing", type=click.BOOL, default=False, is_flag=True,
              help="Only rewrite missing images.")
@click.option("--noswap", type=click.BOOL, default=False, is_flag=True,
              help="Do not produce side-swapped samples.")
@click.option("--core_joints", type=click.BOOL, default=False, is_flag=True,
              help="Use only the 14 usual joints projected from SMPL.")
@click.option("--human_annotations", type=click.BOOL, default=False, is_flag=True,
              help="Use the human annotations (FashionPose should be excluded!).")
@click.option("--up3d_fp", type=click.Path(file_okay=False, readable=True),
              default=UP3D_FP,
              help="Path to the UP3D folder that you want to use.")
def cli(suffix, target_person_size, crop=513, test_crop=513,  # pylint: disable=too-many-locals, too-many-arguments
        only_missing=False, noswap=False, core_joints=False,
        human_annotations=False, up3d_fp=UP3D_FP):
    """Create segmentation datasets from select SMPL fits."""
    np.random.seed(1)
    with_rlswap = not noswap
    if human_annotations:
        assert core_joints
    if test_crop < target_person_size or crop < target_person_size:
        LOGGER.critical("Too small crop size!")
        raise Exception("Too small crop size!")
    landmark_mapping = landmark_mesh_91
    if core_joints:
        LOGGER.info("Using the core joints.")
        # Order is important here! This way, we maintain LSP compatibility.
        landmark_mapping = OrderedDict([('neck', landmark_mapping['neck']),
                                        ('head_top', landmark_mapping['head_top']),])
    n_landmarks = len(landmark_mapping) + 12
    LOGGER.info("Creating pose dataset with %d landmarks with target "
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
    LOGGER.info("Creating list files...")
    list_fp = path.join(path.dirname(__file__), '..', 'training', 'list')
    if not path.exists(list_fp):
        os.makedirs(list_fp)
    train_list_f = open(path.join(list_fp, 'train_%d_%s_%s.txt' % (
        n_landmarks, target_person_size, suffix)), 'w')
    val_list_f = open(path.join(list_fp, 'val_%d_%s_%s.txt' % (
        n_landmarks, target_person_size, suffix)), 'w')
    train_val_list_f = open(path.join(list_fp, 'trainval_%d_%s_%s.txt' % (
        n_landmarks, target_person_size, suffix)), 'w')
    test_list_f = open(path.join(list_fp, 'test_%d_%s_%s.txt' % (
        n_landmarks, target_person_size, suffix)), 'w')
    scale_f = open(path.join(list_fp, 'scale_%d_%s_%s.txt' % (
        n_landmarks, target_person_size, suffix)), 'w')
    with open(path.join(up3d_fp, 'train.txt'), 'r') as f:
        train_spec = [line.strip() for line in f.readlines()]
    with open(path.join(up3d_fp, 'val.txt'), 'r') as f:
        val_spec = [line.strip() for line in f.readlines()]
    with open(path.join(up3d_fp, 'test.txt'), 'r') as f:
        test_spec = [line.strip() for line in f.readlines()]

    LOGGER.info("Processing...")
    list_ids = np.zeros((4,), dtype='int')
    add_dataset(
        dset_fp,
        dset_fromroot,
        list_ids,
        up3d_fp,
        train_list_f,
        val_list_f,
        train_val_list_f,
        test_list_f,
        scale_f,
        train_spec,
        val_spec,
        test_spec,
        target_person_size, landmark_mapping,
        crop, test_crop, 0,
        only_missing=only_missing,
        with_rlswap=with_rlswap,
        human_annotations=human_annotations)
    train_list_f.close()
    val_list_f.close()
    train_val_list_f.close()
    test_list_f.close()
    scale_f.close()    
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("opendr.lighting").setLevel(logging.WARN)
    cli()  # pylint: disable=no-value-for-parameter

