#!/usr/bin/env python2
"""Create segmentation datasets from select SMPL fits."""
import os
import os.path as path
import sys
import logging

import numpy as np
import scipy
import click
import tqdm

from clustertools.log import LOGFORMAT
from clustertools.visualization import apply_colormap
from up_tools.model import (robust_person_size, six_region_groups,
                            regions_to_classes, get_crop)

from up_tools.render_segmented_views import render_body_impl
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))
from config import SEG_DATA_FP, UP3D_FP


LOGGER = logging.getLogger(__name__)
DSET_ROOT_FP = SEG_DATA_FP

if not path.exists(DSET_ROOT_FP):
    os.mkdir(DSET_ROOT_FP)


def uncrop(annot, fullimsize, cropinfo):
    if annot.ndim == 2:
        res = np.zeros((fullimsize[0], fullimsize[1]), dtype='uint8')
    else:
        res = np.ones((fullimsize[0], fullimsize[1], 3), dtype='uint8') * 255
    res[cropinfo[2]:cropinfo[3],
        cropinfo[4]:cropinfo[5]] = scipy.misc.imresize(
            annot,
            (cropinfo[3] - cropinfo[2],
             cropinfo[5] - cropinfo[4]),
            interp='nearest')
    return res


def add_dataset(dset_fp, dset_rel_fp, up3d_fp,  # pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-branches
                train_list_f, val_list_f, test_list_f,
                train_spec, val_spec, test_spec,
                target_person_size, partspec, crop, running_idx,
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
        assert image.ndim == 3
        out_exists = (path.exists(path.join(dset_fp, '%05d_image.png' % (running_idx))) and
                      path.exists(path.join(dset_fp, '%05d_ann.png' % (running_idx))) and
                      path.exists(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx))) and
                      path.exists(path.join(dset_fp, '%05d_render.png' % (running_idx))) and
                      path.exists(path.join(dset_fp, '%05d_render_light.png' % (running_idx))))
        if not (only_missing and out_exists):
            rendering = uncrop(render_body_impl(path.join(up3d_fp, '%05d_body.pkl' % (im_idx)),
                                                resolution=(cropinfo[1],
                                                            cropinfo[0]),
                                                quiet=True,
                                                use_light=False)[0],
                               image.shape[:2],
                               cropinfo)
            rendering_l = uncrop(render_body_impl(path.join(up3d_fp, '%05d_body.pkl' % (im_idx)),
                                                  resolution=(cropinfo[1],
                                                              cropinfo[0]),
                                                  quiet=True,
                                                  use_light=True)[0],
                                 image.shape[:2],
                                 cropinfo)
        joints = np.load(path.join(up3d_fp, '%05d_joints.npy' % (im_idx)))
        joints = np.vstack((joints, np.all(joints > 0, axis=0)[None, :]))
        person_size = robust_person_size(joints)
        norm_factor = float(target_person_size) / person_size
        if not (only_missing and out_exists):
            image = scipy.misc.imresize(image, norm_factor, interp='bilinear')
            rendering = scipy.misc.imresize(rendering, norm_factor, interp='nearest')
            rendering_l = scipy.misc.imresize(rendering_l, norm_factor, interp='bilinear')
            if image.shape[0] > crop or image.shape[1] > crop:
                LOGGER.debug("Image (original %d, here %d) too large (%s)! Cropping...",
                             im_idx, running_idx, str(image.shape[:2]))
                person_center = np.mean(joints[:2, joints[2, :] == 1], axis=1) * norm_factor
                crop_y, crop_x = get_crop(image, person_center, crop)
                image = image[crop_y[0]:crop_y[1],
                              crop_x[0]:crop_x[1], :]
                rendering = rendering[crop_y[0]:crop_y[1],
                                      crop_x[0]:crop_x[1], :]
                rendering_l = rendering_l[crop_y[0]:crop_y[1],
                                          crop_x[0]:crop_x[1], :]
                assert image.shape[0] == crop or image.shape[1] == crop, (
                    "Error cropping image (original %d, here %d)!" % (im_idx,
                                                                      running_idx))
            assert image.shape[0] <= crop and image.shape[1] <= crop and image.shape[2] == 3, (
                "Wrong image shape (original %d, here %d)!" % (im_idx, running_idx))
            class_groups = six_region_groups if partspec == '6' else None
            annotation = regions_to_classes(rendering, class_groups, warn_id=str(im_idx))
            if partspec == '1':
                annotation = (annotation > 0).astype('uint8')
            assert np.max(annotation) <= int(partspec), (
                "Wrong annotation value (original %d, here %d): %s!" % (
                    im_idx, running_idx, str(np.unique(annotation))))
            if running_idx == 0:
                assert np.max(annotation) == int(partspec), (
                    "Probably an error in the number of parts!")
            scipy.misc.imsave(path.join(dset_fp, '%05d_image.png' % (running_idx)), image)
            scipy.misc.imsave(path.join(dset_fp, '%05d_ann.png' % (running_idx)), annotation)
            scipy.misc.imsave(path.join(dset_fp, '%05d_ann_vis.png' % (running_idx)),
                              apply_colormap(annotation, vmax=int(partspec)))
            scipy.misc.imsave(path.join(dset_fp, '%05d_render.png' % (running_idx)), rendering)
            scipy.misc.imsave(path.join(dset_fp, '%05d_render_light.png' % (running_idx)), rendering_l)  # pylint: disable=line-too-long
        if im_idx in train_ids:
            list_f = train_list_f
        elif im_idx in val_ids:
            list_f = val_list_f
        elif im_idx in test_ids:
            list_f = test_list_f
        list_f.write("/%s/%05d_image.png /%s/%05d_ann.png %f\n" % (
            dset_rel_fp, running_idx, dset_rel_fp, running_idx, norm_factor))
        list_f.flush()
        running_idx += 1
    return running_idx


@click.command()
@click.argument("suffix", type=click.STRING)
@click.argument("partspec", type=click.Choice(['1', '6', '31']))
@click.argument("target_person_size", type=click.INT)
@click.option("--crop", type=click.INT, default=513,
              help="Crop size for the images.")
@click.option("--only_missing", type=click.BOOL, default=False, is_flag=True,
              help="Only rewrite missing images.")
@click.option("--up3d_fp", type=click.Path(file_okay=False, readable=True),
              default=UP3D_FP,
              help="Path to the UP3D folder that you want to use.")
def cli(suffix, partspec, target_person_size, crop=513, only_missing=False, up3d_fp=UP3D_FP):  # pylint: disable=too-many-locals, too-many-arguments
    """Create segmentation datasets from select SMPL fits."""
    np.random.seed(1)
    LOGGER.info("Creating segmentation dataset for %s classes with target "
                "person size %f and suffix `%s`.",
                partspec, target_person_size, suffix)
    assert ' ' not in suffix
    dset_fromroot = path.join(partspec, str(target_person_size), suffix)
    dset_fp = path.join(DSET_ROOT_FP, dset_fromroot)
    if path.exists(dset_fp):
        if not only_missing:
            if not click.confirm("Dataset folder exists: `%s`! Continue?" % (dset_fp)):
                return
    else:
        os.makedirs(dset_fp)
    LOGGER.info("Creating list files...")
    list_fp = path.join(path.dirname(__file__), '..', 'training', 'list')
    if not path.exists(list_fp):
        os.makedirs(list_fp)
    train_list_f = open(path.join(list_fp, 'train_%s_%d_%s.txt' % (
        partspec, target_person_size, suffix)), 'w')
    val_list_f = open(path.join(list_fp, 'val_%s_%d_%s.txt' % (
        partspec, target_person_size, suffix)), 'w')
    test_list_f = open(path.join(list_fp, 'test_%s_%d_%s.txt' % (
        partspec, target_person_size, suffix)), 'w')
    with open(path.join(up3d_fp, 'train.txt'), 'r') as f:
        train_spec = [line.strip() for line in f.readlines()]
    with open(path.join(up3d_fp, 'val.txt'), 'r') as f:
        val_spec = [line.strip() for line in f.readlines()]
    with open(path.join(up3d_fp, 'test.txt'), 'r') as f:
        test_spec = [line.strip() for line in f.readlines()]

    LOGGER.info("Processing...")
    add_dataset(
        dset_fp,
        dset_fromroot,
        up3d_fp,
        train_list_f, val_list_f, test_list_f,
        train_spec, val_spec, test_spec,
        target_person_size, partspec,
        crop, 0,
        only_missing=only_missing)
    train_list_f.close()
    val_list_f.close()
    test_list_f.close()
    LOGGER.info("Creating trainval file...")
    trainval_list_f = open(path.join(list_fp, 'trainval_%s_%d_%s.txt' % (
        partspec, target_person_size, suffix)), 'w')
    train_list_f = open(path.join(list_fp, 'train_%s_%d_%s.txt' % (
        partspec, target_person_size, suffix)), 'r')
    val_list_f = open(path.join(list_fp, 'val_%s_%d_%s.txt' % (
        partspec, target_person_size, suffix)), 'r')
    for line in train_list_f:
        trainval_list_f.write(line)
    for line in val_list_f:
        trainval_list_f.write(line)
    trainval_list_f.close()
    train_list_f.close()
    val_list_f.close()
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("opendr.lighting").setLevel(logging.WARN)
    cli()  # pylint: disable=no-value-for-parameter
