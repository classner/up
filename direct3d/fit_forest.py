#!/usr/bin/env python2
"""Forest training."""
# pylint: disable=wrong-import-order, redefined-outer-name, line-too-long, invalid-name
import os
import sys
import os.path as path
import h5py
import numpy as np
import sys
import logging
import click
from up_tools.model import landmarks_91
import joblib
import pyximport; pyximport.install()  # pylint: disable=multiple-statements
from conversions import (  # pylint: disable=import-error
    axis_angle_to_matrix)
from clustertools.config import available_cpu_count
from clustertools.log import LOGFORMAT

sys.path.insert(0, path.join(path.dirname(__file__),
                             '..'))
from config import DIRECT3D_DATA_FP

LOGGER = logging.getLogger(__name__)
OUT_DIR = path.join(path.dirname(__file__),
                    '..', 'models', '2dto3d',
                    'separate_regressors')
if not path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
# Inputs:
# * 2x91 absolute x, y coordinates
# Prediction targets:
# * betas: 10
# * pose: 72
# * trans: 3
# * rt: 3
# * t: 3
# * f: 1

torso_ids = [landmarks_91.rshoulder,
             landmarks_91.rshoulder_back,
             landmarks_91.rshoulder_front,
             landmarks_91.rshoulder_back,

             landmarks_91.lshoulder,
             landmarks_91.lshoulder_back,
             landmarks_91.lshoulder_front,
             landmarks_91.lshoulder_top,

             landmarks_91.shoulderblade_center,
             landmarks_91.solar_plexus,
             landmarks_91.lpapilla,
             landmarks_91.rpapilla,

             landmarks_91.belly_button,
             landmarks_91.rwaist,
             landmarks_91.lwaist,
             landmarks_91.waist_back,

             landmarks_91.lhip,
             landmarks_91.lhip_back,
             landmarks_91.lhip_front,
             landmarks_91.lhip_outer,
             landmarks_91.rhip,
             landmarks_91.rhip_back,
             landmarks_91.rhip_front,
             landmarks_91.rhip_outer]

head_ids = [landmarks_91.head_top,
            landmarks_91.head_back,
            landmarks_91.nose,
            landmarks_91.lear,
            landmarks_91.rear]

larm_ids = [landmarks_91.luarm_inner,
            landmarks_91.luarm_outer,
            landmarks_91.lelbow,
            landmarks_91.lelbow_bottom,
            landmarks_91.lelbow_inner,
            landmarks_91.lelbow_outer,
            landmarks_91.lelbow_top,
            landmarks_91.llarm_lower,
            landmarks_91.llarm_upper,
            landmarks_91.lwrist]

rarm_ids = [landmarks_91.ruarm_inner,
            landmarks_91.ruarm_outer,
            landmarks_91.relbow,
            landmarks_91.relbow_bottom,
            landmarks_91.relbow_inner,
            landmarks_91.relbow_outer,
            landmarks_91.relbow_top,
            landmarks_91.rlarm_lower,
            landmarks_91.rlarm_upper,
            landmarks_91.rwrist]

lleg_ids = [landmarks_91.luleg_back,
            landmarks_91.luleg_front,
            landmarks_91.luleg_outer,
            landmarks_91.luleg_inner,
            landmarks_91.lknee,
            landmarks_91.llleg_back,
            landmarks_91.llleg_front,
            landmarks_91.llleg_outer,
            landmarks_91.llleg_inner,
            landmarks_91.lankle,
            landmarks_91.lheel,
            landmarks_91.lankle_inner,
            landmarks_91.lankle_outer,
            landmarks_91.lbigtoe]

rleg_ids = [landmarks_91.ruleg_back,
            landmarks_91.ruleg_front,
            landmarks_91.ruleg_outer,
            landmarks_91.ruleg_inner,
            landmarks_91.rknee,
            landmarks_91.rlleg_back,
            landmarks_91.rlleg_front,
            landmarks_91.rlleg_outer,
            landmarks_91.rlleg_inner,
            landmarks_91.rankle,
            landmarks_91.rheel,
            landmarks_91.rankle_inner,
            landmarks_91.rankle_outer,
            landmarks_91.rbigtoe]

def create_featseltuple(ids):
    """Add the flat y coordinate to an index list."""
    newlist = []
    for part_id in ids:
        newlist.extend([part_id, part_id + 91])
    return tuple(sorted(newlist))

lmset_to_use = {
    (0, 10): torso_ids,  # shape
    (10, 13): torso_ids,  # root joint
    (13, 16): torso_ids + lleg_ids,  # luleg
    (16, 19): torso_ids + rleg_ids,  # ruleg
    (19, 22): torso_ids,  # spine
    (22, 25): lleg_ids,  # llleg
    (25, 28): rleg_ids,  # rlleg
    (28, 31): torso_ids,  # spine1
    (31, 34): lleg_ids,  # lfoot
    (34, 37): rleg_ids,  # rfoot
    (37, 40): torso_ids,  # spine2
    (40, 43): lleg_ids,  # ltoes
    (43, 46): rleg_ids,  # rtoes
    (46, 49): torso_ids + head_ids,  # neck
    (49, 52): torso_ids + larm_ids,  # lshoulder
    (52, 55): torso_ids + rarm_ids,  # rshoulder
    (55, 58): torso_ids + head_ids,  # head
    (58, 61): torso_ids + larm_ids,  # luarm
    (61, 64): torso_ids + rarm_ids,  # ruarm
    (64, 67): larm_ids,  # llarm
    (67, 70): rarm_ids,  # rlarm
    (70, 73): larm_ids,  # lhand
    (73, 76): rarm_ids,  # rhand
    (76, 79): larm_ids,  # lfingers
    (79, 82): rarm_ids,  # rfingers
    (82, 85): torso_ids  # depth
    }



def normalize_axis_angle(anglevec):
    """Normalize angle periodicity."""
    assert len(anglevec) % 3 == 0
    for startpos in range(0, len(anglevec), 3):
        rep = anglevec[startpos:startpos + 3]
        angle = np.linalg.norm(rep)
        angle_norm = np.fmod(angle, np.pi)
        anglevec[startpos:startpos + 3] = rep / angle * angle_norm

def preprocess(dta_arr):
    """Make the coordinates relative to mean position, apply the modulo operator to pose."""
    for dta_idx in range(dta_arr.shape[0]):
        pose = dta_arr[dta_idx, :2*91].reshape((2, 91))
        mean = np.mean(pose, axis=1)
        dta_arr[dta_idx, :2*91] = (pose.T - mean + 513. / 2.).T.flat
        normalize_axis_angle(dta_arr[dta_idx, 2*91+10:2*91+10+72])

def get_data(prefix, part_rangestart, finalize, debug_run):  # pylint: disable=too-many-branches
    """Get the data."""
    rangestart = part_rangestart
    rangeend = 10 if part_rangestart == 0 else part_rangestart + 3
    train_f = h5py.File(path.join(
        DIRECT3D_DATA_FP,
        '91', '500', prefix, 'train.hdf5'))
    train_dset = train_f['2dto3d']
    if debug_run:
        train_dta = np.array(train_dset[:10000])
    else:
        train_dta = np.array(train_dset)
    preprocess(train_dta)
    #add_noise(train_dta)
    val_f = h5py.File(path.join(
        DIRECT3D_DATA_FP,
        '91', '500', prefix, 'val.hdf5'))
    val_dset = val_f['2dto3d']
    if debug_run:
        val_dta = np.array(val_dset[:10])
    else:
        val_dta = np.array(val_dset)
    preprocess(val_dta)
    if finalize:
        train_dta = np.vstack((train_dta, val_dta))
        val_f = h5py.File(path.join(
        DIRECT3D_DATA_FP,
        '91', '500', prefix, 'test.hdf5'))
        val_dset = val_f['2dto3d']
        if debug_run:
            val_dta = np.array(val_dset[:10])
        else:
            val_dta = np.array(val_dset)
        preprocess(val_dta)
    train_annot = train_dta[:, 182+rangestart:182+rangeend]
    val_annot = val_dta[:, 182+rangestart:182+rangeend]
    rel_ids = create_featseltuple(lmset_to_use[(rangestart, rangeend)])
    train_dta = train_dta[:, rel_ids]
    val_dta = val_dta[:, rel_ids]
    if rangestart > 0 and rangestart < 82:
        train_annot = axis_angle_to_matrix(train_annot)
        val_annot = axis_angle_to_matrix(val_annot)
    return train_dta, train_annot, val_dta, val_annot

def sqdiff(rnge, val_dta, val_results, addoffs=0):
    """Error measure robust to angle orientations and mirroring."""
    orig_ids = tuple(np.array(rnge) + 182 + addoffs)
    val_ids = tuple(np.array(rnge))
    assert len(orig_ids) == len(val_ids)
    assert len(orig_ids) % 3 == 0
    diffs = []
    for sample_idx in range(val_dta.shape[0]):
        for rot_idx in range(0, len(orig_ids), 3):
            plaindiff = np.linalg.norm(val_dta[sample_idx, orig_ids[rot_idx:rot_idx+3]] -
                                       val_results[sample_idx, val_ids[rot_idx:rot_idx+3]])
            mirrdiff = np.linalg.norm(val_dta[sample_idx, orig_ids[rot_idx:rot_idx+3]] +
                                      val_results[sample_idx, val_ids[rot_idx:rot_idx+3]])
            diffs.append(min(plaindiff, mirrdiff))
    return np.mean(diffs)

@click.command()
@click.argument('train_prefix', type=click.STRING)
@click.argument('part_rangestart', type=click.INT)
@click.option('--finalize', type=click.BOOL, default=False, is_flag=True,
              help='Train on train+val, test on test.')
@click.option('--debug_run', type=click.BOOL, default=False, is_flag=True,
              help='Use only a small fraction of data for testing.')
def cli(train_prefix, part_rangestart,  # pylint: disable=too-many-branches, too-many-locals, too-many-statements, too-many-arguments
        finalize=False, debug_run=False):
    """Run a RotatingTree experiment."""
    rangestart = part_rangestart
    pref = 'forest'
    pref += '_' + str(part_rangestart)
    if finalize:
        pref += '_final'
    if debug_run:
        pref += '_debug'
    out_fp = path.join(OUT_DIR, pref + '.z')
    LOGGER.info("Running for configuration `%s`.", out_fp)
    LOGGER.info("Loading data...")
    train_dta, train_annot, val_dta, val_annot = get_data(  # pylint: disable=unused-variable
        train_prefix, part_rangestart, finalize, debug_run)
    # Checks.
    if rangestart > 0 and rangestart < 82:
        # Rotation matrices.
        assert train_annot.max() <= 1.
        assert train_annot.min() >= -1.
        assert val_annot.max() <= 1.
        assert val_annot.min() >= -1.
    import sklearn.ensemble
    rf = sklearn.ensemble.RandomForestRegressor(n_jobs=available_cpu_count())
    LOGGER.info("Fitting...")
    rf.fit(train_dta, train_annot)
    LOGGER.info("Writing results...")
    joblib.dump(rf, out_fp, compress=True)
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()  # pylint: disable=no-value-for-parameter
