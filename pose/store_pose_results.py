#!/usr/bin/env python2
"""Write the pose results to disk for further evaluation."""
# pylint: disable=invalid-name
from os import path
import sys
import logging
from PIL import Image

import numpy as np
import scipy.misc
import scipy.misc as sm
import click
import tqdm

from up_tools.model import (
    connections_landmarks_91, dots_landmarks_91,
    reduction_91tolsp)
from up_tools.visualization import visualize_pose
from up_tools.model import connections_lsp
LOGGER = logging.getLogger(__name__)

# Image mean to use.
_MEAN = np.array([104., 117., 123.])
# Scale factor for the CNN offset predictions.
_LOCREF_SCALE_MUL = np.sqrt(53.)
# Maximum size of one tile to process (to limit the required GPU memory).
_MAX_SIZE = 700

_STRIDE = 8.

# CNN model store.
_MODEL = None


def estimate_pose(image, model_def, model_bin, caffe, scales=None):  # pylint: disable=too-many-locals
    """
    Get the estimated pose for an image.

    Uses the CNN pose estimator from "Deepcut: Joint Subset Partition and
    Labeling for Multi Person Pose Estimation" (Pishchulin et al., 2016),
    "DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation
    Model" (Insafutdinov et al., 2016).


    Parameters
    ==========

    :param image: np.array(3D).
      The image in height X width X BGR format.

    :param scales: list(float) or None.
      Scales on which to apply the estimator. The pose with the best confidence
      on average will be returned.

    Returns
    =======

    :param pose: np.array(2D).
      The pose in 5x14 layout. The first axis is along per-joint information,
      the second the joints. Information is:
        1. position x,
        2. position y,
        3. CNN confidence,
        4. CNN offset vector x,
        5. CNN offset vector y.
    """
    global _MODEL  # pylint: disable=global-statement
    if scales is None:
        scales = [1.]
    if _MODEL is None:
        LOGGER.info("Loading pose model...")
        _MODEL = caffe.Net(model_def, model_bin, caffe.TEST)
        LOGGER.info("Done!")
    LOGGER.debug("Processing image...")
    im_orig = image.copy()
    LOGGER.debug("Image shape: %s.", im_orig.shape)
    best_pose = None
    highest_confidence = 0.
    for scale_factor in scales:
        LOGGER.debug("Scale %f...", scale_factor)
        image = im_orig.copy()
        # Create input for multiples of net input/output changes.
        im_bg_width = int(np.ceil(
            float(image.shape[1]) * scale_factor / _STRIDE) * _STRIDE)
        im_bg_height = int(np.ceil(
            float(image.shape[0]) * scale_factor / _STRIDE) * _STRIDE)
        pad_size = 64
        im_bot_pixels = image[-1:, :, :]
        im_bot = np.tile(im_bot_pixels, (pad_size, 1, 1))
        image = np.vstack((image, im_bot))
        im_right_pixels = image[:, -1:, :]
        im_right = np.tile(im_right_pixels, (1, pad_size, 1))
        image = np.hstack((image, im_right))
        image = scipy.misc.imresize(image, scale_factor, interp='bilinear')
        image = image.astype('float32') - _MEAN

        net_input = np.zeros((im_bg_height, im_bg_width, 3), dtype='float32')
        net_input[:min(net_input.shape[0], image.shape[0]),
                  :min(net_input.shape[1], image.shape[1]), :] =\
            image[:min(net_input.shape[0], image.shape[0]),
                  :min(net_input.shape[1], image.shape[1]), :]

        LOGGER.debug("Input shape: %d, %d.",
                     net_input.shape[0], net_input.shape[1])
        unary_maps, locreg_pred = _process_image_tiled(_MODEL, net_input, _STRIDE)

        """
        import matplotlib.pyplot as plt
        plt.figure()
        for map_idx in range(unary_maps.shape[2]):
            plt.imshow(unary_maps[:, :, map_idx], interpolation='none')
            plt.imsave('map_%d.png' % map_idx,
                       unary_maps[:, :, map_idx])
            plt.show()
        """

        pose = _pose_from_mats(unary_maps, locreg_pred, scale=scale_factor)

        minconf = np.min(pose[2, :])
        if minconf > highest_confidence:
            LOGGER.debug("New best scale detected: %f (scale), " +
                         "%f (min confidence).", scale_factor, minconf)
            highest_confidence = minconf
            best_pose = pose
    LOGGER.debug("Pose estimated.")
    return best_pose


def _pose_from_mats(scoremat, offmat, scale):
    """Combine scoremat and offsets to the final pose."""
    pose = []
    for joint_idx in range(scoremat.shape[2]):
        maxloc = np.unravel_index(np.argmax(scoremat[:, :, joint_idx]),
                                  scoremat[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = (np.array(maxloc).astype('float') * _STRIDE + 0.5 * _STRIDE +
                  offset * _LOCREF_SCALE_MUL)
        pose.append(np.hstack((pos_f8[::-1] / scale,
                               [scoremat[maxloc][joint_idx]],
                               offset * _LOCREF_SCALE_MUL / scale)))
    return np.array(pose).T


def _get_num_tiles(length, max_size, rf):
    """Get the number of tiles required to cover the entire image."""
    if length <= max_size:
        return 1
    k = 0
    while True:
        new_size = (max_size - rf) * 2 + (max_size - 2*rf) * k
        if new_size > length:
            break
        k += 1
    return 2 + k


# pylint: disable=too-many-locals
def _process_image_tiled(model, net_input, stride):
    """Get the CNN results for the tiled image."""
    rf = 224  # Standard receptive field size.
    cut_off = rf / stride

    num_tiles_x = _get_num_tiles(net_input.shape[1], _MAX_SIZE, rf)
    num_tiles_y = _get_num_tiles(net_input.shape[0], _MAX_SIZE, rf)
    if num_tiles_x > 1 or num_tiles_y > 1:
        LOGGER.info("Tiling the image into %d, %d (w, h) tiles...",
                    num_tiles_x, num_tiles_y)

    scoremaps = []
    locreg_pred = []
    for j in range(num_tiles_y):
        start_y = j * (_MAX_SIZE - 2*rf)
        if j == num_tiles_y:
            end_y = net_input.shape[0]
        else:
            end_y = start_y + _MAX_SIZE
        scoremaps_line = []
        locreg_pred_line = []
        for i in range(num_tiles_x):
            start_x = i * (_MAX_SIZE - 2*rf)
            if i == num_tiles_x:
                end_x = net_input.shape[1]
            else:
                end_x = start_x + _MAX_SIZE
            input_tile = net_input[start_y:end_y, start_x:end_x, :]
            LOGGER.debug("Tile info: %d, %d, %d, %d.",
                         start_y, end_y, start_x, end_x)
            scoremaps_tile, locreg_pred_tile = _cnn_process_image(model,
                                                                  input_tile)
            LOGGER.debug("Tile out shape: %s, %s.",
                         str(scoremaps_tile.shape),
                         str(locreg_pred_tile.shape))
            scoremaps_tile = _cutoff_tile(scoremaps_tile,
                                          num_tiles_x, i, cut_off, True)
            locreg_pred_tile = _cutoff_tile(locreg_pred_tile,
                                            num_tiles_x, i, cut_off, True)
            scoremaps_tile = _cutoff_tile(scoremaps_tile,
                                          num_tiles_y, j, cut_off, False)
            locreg_pred_tile = _cutoff_tile(locreg_pred_tile,
                                            num_tiles_y, j, cut_off, False)
            LOGGER.debug("Cutoff tile out shape: %s, %s.",
                         str(scoremaps_tile.shape), str(locreg_pred_tile.shape))
            scoremaps_line.append(scoremaps_tile)
            locreg_pred_line.append(locreg_pred_tile)
        scoremaps_line = np.concatenate(scoremaps_line, axis=1)
        locreg_pred_line = np.concatenate(locreg_pred_line, axis=1)
        scoremaps_line = _cutoff_tile(scoremaps_line,
                                      num_tiles_y, j, cut_off, False)
        locreg_pred_line = _cutoff_tile(locreg_pred_line,
                                        num_tiles_y, j, cut_off, False)
        LOGGER.debug("Line tile out shape: %s, %s.",
                     str(scoremaps_line.shape), str(locreg_pred_line.shape))
        scoremaps.append(scoremaps_line)
        locreg_pred.append(locreg_pred_line)
    scoremaps = np.concatenate(scoremaps, axis=0)
    locreg_pred = np.concatenate(locreg_pred, axis=0)
    LOGGER.debug("Final tiled shape: %s, %s.",
                 str(scoremaps.shape), str(locreg_pred.shape))
    return scoremaps[:, :, 0, :], locreg_pred.transpose((0, 1, 3, 2))


def _cnn_process_image(model, net_input):
    """Get the CNN results for a fully prepared image."""
    net_input = net_input.transpose((2, 0, 1))
    model.blobs['data'].reshape(1, 3, net_input.shape[1], net_input.shape[2])
    model.blobs['data'].data[0, ...] = net_input[...]
    model.forward()

    out_value = model.blobs['prob'].data
    feat_prob = out_value.copy().transpose((2, 3, 0, 1))
    n_joints = feat_prob.shape[3]
    out_value = model.blobs['loc_pred'].data
    out_value = out_value.reshape((n_joints, 2,
                                   out_value.shape[2],
                                   out_value.shape[3]))
    locreg_pred = out_value.transpose((2, 3, 1, 0))
    return feat_prob, locreg_pred


def _cutoff_tile(sm, num_tiles, idx, cut_off, is_x):
    """Cut the valid parts of the CNN predictions for a tile."""
    if is_x:
        sm = sm.transpose((1, 0, 2, 3))
    if num_tiles == 1:
        pass
    elif idx == 0:
        sm = sm[:-cut_off, ...]
    elif idx == num_tiles-1:
        sm = sm[cut_off:, ...]
    else:
        sm = sm[cut_off:-cut_off, ...]
    if is_x:
        sm = sm.transpose((1, 0, 2, 3))
    return sm


@click.command()
@click.argument("caffe_prototxt", type=click.Path(dir_okay=False, readable=True))
@click.argument("caffe_model", type=click.Path(dir_okay=False, readable=True))
@click.argument("image_list_file", type=click.Path(dir_okay=False, readable=True))
@click.argument("dset_root", type=click.Path(file_okay=False, readable=True))
@click.argument("output_folder", type=click.Path(file_okay=False, writable=True))
@click.argument("caffe_install_path", type=click.Path(file_okay=False, readable=True))
def main(caffe_prototxt,  # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
         caffe_model,
         image_list_file,
         dset_root,
         output_folder,
         caffe_install_path):
    """Store and visualize the pose results for a model."""
    LOGGER.info("Storing pose results to folder `%s`.", output_folder)
    LOGGER.info("Using caffe from `%s`.", caffe_install_path)
    sys.path.insert(0, path.join(caffe_install_path))
    import caffe  # pylint: disable=import-error
    caffe.set_mode_gpu()
    with open(image_list_file, 'r') as inf:
        image_list = inf.readlines()
    image_list = [path.join(dset_root, line[1:])
                  for line in image_list if line.startswith('/')]
    n_landmarks = int(image_list_file[image_list_file.find('_')+1:
                                      image_list_file.find('_')+3])
    for imgnames in tqdm.tqdm(image_list):
        imgname = imgnames.split(" ")[0].strip()
        LOGGER.debug("Processing `%s`...", imgname)
        im = caffe.io.load_image(imgname)  # pylint: disable=invalid-name
        # caffe.io loads as RGB, and in range [0., 1.].
        im = (im * 255.)[:, :, ::-1].astype('uint8')
        landmark_locs = estimate_pose(im, caffe_prototxt, caffe_model, caffe)
        if landmark_locs.shape[1] == 91 and n_landmarks == 14:
            # Extract LSP joints.
            landmark_locs = landmark_locs[:, reduction_91tolsp]
        np.save(path.join(output_folder,
                          path.basename(imgname) + '.npy'),
                landmark_locs)
        vis_im = visualize_pose(im[:, :, ::-1],
                                landmark_locs, scale=1., dash_length=5)
        sm.imsave(path.join(output_folder,
                            path.basename(imgname) + '.npy.vis.png'),
                  vis_im)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
