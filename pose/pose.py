#!/usr/bin/env python2
"""
Pose predictions in Python.

Caffe must be available on the Pythonpath for this to work. The methods can
be imported and used directly, or the command line interface can be used. In
the latter case, adjust the log-level to your needs. The maximum image size
for one prediction can be adjusted with the variable _MAX_SIZE so that it
still fits in GPU memory, all larger images are split in sufficiently small
parts.

Authors: Christoph Lassner, based on the MATLAB implementation by Eldar
  Insafutdinov.
"""
# pylint: disable=invalid-name, wrong-import-position
import os as _os
import os.path as _path
import logging as _logging
import glob as _glob
import sys

import numpy as _np
import scipy as _scipy
import click as _click
sys.path.insert(0, _path.join(_path.dirname(__file__), '..'))
from config import DEEPERCUT_CNN_BUILD_FP
sys.path.insert(0, _path.join(DEEPERCUT_CNN_BUILD_FP, 'install', 'python'))
import caffe as _caffe  # pylint: disable=import-error
from up_tools.visualization import visualize_pose  # pylint: disable=import-error
try:
    import cv2 as _cv2  # pylint: disable=wrong-import-order
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
from clustertools.log import LOGFORMAT


_LOGGER = _logging.getLogger(__name__)

# Constants.
# Image mean to use.
_MEAN = _np.array([104., 117., 123.])
# Scale factor for the CNN offset predictions.
_LOCREF_SCALE_MUL = _np.sqrt(53.)
# Maximum size of one tile to process (to limit the required GPU memory).
_MAX_SIZE = 700  # Used to be 700

# CNN model store.
_MODEL = None


def estimate_pose(image, scales=None):  # pylint: disable=too-many-locals
    """
    Get the estimated pose for an image.

    Uses the CNN pose estimator from "Deepcut: Joint Subset Partition and
    Labeling for Multi Person Pose Estimation" (Pishchulin et al., 2016).

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
        _LOGGER.info("Loading pose model...")
        _MODEL = _caffe.Net(
            _os.path.join(_os.path.abspath(_os.path.dirname(__file__)),
                          '..',
                          'models',
                          'pose',
                          'testpy_val_91_500_pkg.prototxt'),
            _os.path.join(_os.path.abspath(_os.path.dirname(__file__)),
                          'training',
                          'model',
                          'pose',
                          'test2.caffemodel'),
            _caffe.TEST)
        _LOGGER.info("Done!")
    _LOGGER.debug("Processing image...")
    im_orig = image.copy()
    _LOGGER.debug("Image shape: %s.", im_orig.shape)
    best_pose = None
    highest_confidence = 0.
    pose_versions = []
    for scale_factor in scales:
        _LOGGER.debug("Scale %f...", scale_factor)
        image = im_orig.copy()
        # Create input for multiples of net input/output changes.
        im_bg_width = int(_np.ceil(
            float(image.shape[1]) * scale_factor / 8.) * 8.)
        im_bg_height = int(_np.ceil(
            float(image.shape[0]) * scale_factor / 8.) * 8.)
        pad_size = 64
        im_bot_pixels = image[-1:, :, :]
        im_bot = _np.tile(im_bot_pixels, (pad_size, 1, 1))
        image = _np.vstack((image, im_bot))
        im_right_pixels = image[:, -1:, :]
        im_right = _np.tile(im_right_pixels, (1, pad_size, 1))
        image = _np.hstack((image, im_right))
        image = _scipy.misc.imresize(image, scale_factor, interp='bilinear')
        image = image.astype('float32') - _MEAN

        net_input = _np.zeros((im_bg_height, im_bg_width, 3), dtype='float32')
        net_input[:min(net_input.shape[0], image.shape[0]),
                  :min(net_input.shape[1], image.shape[1]), :] =\
            image[:min(net_input.shape[0], image.shape[0]),
                  :min(net_input.shape[1], image.shape[1]), :]
        _LOGGER.debug("Input shape: %d, %d.",
                      net_input.shape[0], net_input.shape[1])
        unary_maps, locreg_pred = _process_image_tiled(_MODEL, net_input, 8)

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
        meanconf = _np.mean(pose[2, :])
        medianconf = _np.median(pose[2, :])
        minconf = _np.min(pose[2, :])
        pose_versions.append(pose.copy())
        _LOGGER.info("conf stats: %f, %f, %f.",  meanconf, medianconf, minconf)
        if minconf > highest_confidence:
            _LOGGER.debug("New best scale detected: %f (scale), " +
                          "%f (min confidence).", scale_factor, minconf)
            highest_confidence = minconf
            best_pose = pose
            best_pmaps = pmap_from_mats(unary_maps,
                                        locreg_pred,
                                        im_orig.shape[:2],
                                        scale_factor)
            """
            best_pmaps = _np.zeros((im_orig.shape[0],
                                    im_orig.shape[1],
                                    tmp_pmaps.shape[2]), dtype='float32')
            for layer_idx in range(best_pmaps.shape[2]):
                best_pmaps[:, :, layer_idx] = _scipy.misc.imresize(tmp_pmaps[:, :, layer_idx],
                                                                   (best_pmaps.shape[0],
                                                                    best_pmaps.shape[1]),
                                                                   interp='bilinear')
            """
    pose_versions = _np.array(pose_versions)
    versions_to_use = _np.argmax(pose_versions[:, 2, :], axis=0)
    best_pose = _np.zeros_like(pose_versions[0])
    for joint_idx in range(pose_versions[0].shape[1]):
        best_pose[:, joint_idx] = pose_versions[versions_to_use[joint_idx],
                                                :,
                                                joint_idx]
    _LOGGER.debug("Pose estimated.")
    return best_pose, best_pmaps


def pmap_from_mats(scoremat,
                   offmat,
                   shape,
                   scale):
    """Get a probability map from the deepcut representation."""
    result = _np.zeros((shape[0], shape[1], scoremat.shape[2]),
                       dtype='float32')
    for joint_idx in range(14):
        for y_idx in range(scoremat.shape[0]):
            for x_idx in range(scoremat.shape[1]):
                offset = offmat[y_idx, x_idx][joint_idx][::-1]
                pos_f8 = (_np.array([y_idx, x_idx]).astype('float') * 8. +
                          offset * _LOCREF_SCALE_MUL)
                pos = (pos_f8 / scale).astype('int')
                pos[0] = max(0,
                             min(pos[0],
                                 result.shape[0] - 1))
                pos[1] = max(0,
                             min((pos[1],
                                  result.shape[1] - 1)))
                result[pos[0], pos[1], joint_idx] = \
                    max(result[pos[0], pos[1], joint_idx],
                        scoremat[y_idx, x_idx, joint_idx])
    return result


def _pose_from_mats(scoremat, offmat, scale):
    """Combine scoremat and offsets to the final pose."""
    pose = []
    for joint_idx in range(scoremat.shape[2]):
        maxloc = _np.unravel_index(_np.argmax(scoremat[:, :, joint_idx]),
                                   scoremat[:, :, joint_idx].shape)
        offset = _np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = (_np.array(maxloc).astype('float') * 8. + 0.5 * 8. +
                  offset * _LOCREF_SCALE_MUL)
        pose.append(_np.hstack((pos_f8[::-1] / scale,
                                [scoremat[maxloc][joint_idx]],
                                offset * _LOCREF_SCALE_MUL / scale)))
    return _np.array(pose).T


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
        _LOGGER.info("Tiling the image into %d, %d (w, h) tiles...",
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
            _LOGGER.debug("Tile info: %d, %d, %d, %d.",
                          start_y, end_y, start_x, end_x)
            scoremaps_tile, locreg_pred_tile = _cnn_process_image(model,
                                                                  input_tile)
            _LOGGER.debug("Tile out shape: %s, %s.",
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
            _LOGGER.debug("Cutoff tile out shape: %s, %s.",
                          str(scoremaps_tile.shape), str(locreg_pred_tile.shape))
            scoremaps_line.append(scoremaps_tile)
            locreg_pred_line.append(locreg_pred_tile)
        scoremaps_line = _np.concatenate(scoremaps_line, axis=1)
        locreg_pred_line = _np.concatenate(locreg_pred_line, axis=1)
        """
        scoremaps_line = _cutoff_tile(scoremaps_line,
                                      num_tiles_y, j, cut_off, False)
        locreg_pred_line = _cutoff_tile(locreg_pred_line,
                                        num_tiles_y, j, cut_off, False)
        """
        _LOGGER.debug("Line tile out shape: %s, %s.",
                      str(scoremaps_line.shape), str(locreg_pred_line.shape))
        scoremaps.append(scoremaps_line)
        locreg_pred.append(locreg_pred_line)
    scoremaps = _np.concatenate(scoremaps, axis=0)
    locreg_pred = _np.concatenate(locreg_pred, axis=0)
    _LOGGER.debug("Final tiled shape: %s, %s.",
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
    elif idx == num_tiles - 1:
        sm = sm[cut_off:, ...]
    else:
        sm = sm[cut_off:-cut_off, ...]
    if is_x:
        sm = sm.transpose((1, 0, 2, 3))
    return sm


def _npcircle(image, cx, cy, radius, color):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = _np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * 0.4 +
        _np.array(color).astype('float32') * 0.6).astype('uint8')


###############################################################################
# Command line interface.
###############################################################################

@_click.command()
@_click.argument('image_name',
                 type=_click.Path(exists=True, dir_okay=True, readable=True))
@_click.option('--out_name',
               type=_click.Path(dir_okay=True, writable=True),
               help='The result location to use. By default, use `image_name`_pose.npz.',
               default=None)
@_click.option('--scales',
               type=_click.STRING,
               help=('The scales to use, comma-separated. The most confident '
                     'will be stored. Default: 1.'),
               default='1.')
@_click.option('--visualize',
               type=_click.BOOL,
               help='Whether to create a visualization of the pose. Default: True.',
               default=True)
@_click.option('--write_pmaps',
               type=_click.BOOL,
               default=False,
               is_flag=True,
               help='Whether to write probability maps additionaly to the keypoints. Default: False.')  # pylint: disable=line-too-long
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help=('The ending to use for the images to read, if a folder is '
                     'specified. Default: .png.'),
               default='.png')
@_click.option('--use_cpu',
               type=_click.BOOL,
               is_flag=True,
               help='Use CPU instead of GPU for predictions.',
               default=False)
@_click.option('--every_nth',
               type=_click.INT,
               default=1,
               help='Predict only every nth frame. Default: 1.')
def predict_pose_from(image_name,  # pylint: disable=too-many-arguments, too-many-branches
                      out_name=None,
                      scales='1.',
                      visualize=True,
                      write_pmaps=False,
                      folder_image_suffix='.png',
                      use_cpu=False,
                      every_nth=1):
    """
    Load an image file, predict the pose and write it out.

    `IMAGE_NAME` may be an image or a directory, for which all images with
    `folder_image_suffix` will be processed.
    """
    scales = [float(val) for val in scales.split(',')]
    if _os.path.isdir(image_name):
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = sorted(_glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix)))
        images = [im_fp for im_fp in images if not im_fp.endswith('vis.png')]
        images = images[::every_nth]
        process_folder = True
    else:
        images = [image_name]
        process_folder = False
    if use_cpu:
        _caffe.set_mode_cpu()
    else:
        _caffe.set_mode_gpu()
    out_name_provided = out_name
    if process_folder and out_name is not None and not _os.path.exists(out_name):
        _os.mkdir(out_name)
    for image_name in images:
        if out_name_provided is None:
            out_name = image_name + '_pose.npz'
            pmap_out_name = image_name + '_pose_probabilities.npz'
        elif process_folder:
            out_name = _os.path.join(out_name_provided,
                                     _os.path.basename(image_name) + '_pose.npz')
        _LOGGER.info("Predicting the pose on `%s` (saving to `%s`) in best of "
                     "scales %s.", image_name, out_name, scales)
        image = _scipy.misc.imread(image_name)
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, ::-1]
        pose, pmaps = estimate_pose(image, scales)
        _np.savez_compressed(out_name, pose=pose)
        if write_pmaps:
            _np.savez_compressed(pmap_out_name, pose_probabilities=pmaps)
        if visualize:
            visim = visualize_pose(image[:, :, ::-1],
                                   pose,
                                   dash_length=5,
                                   opacity=0.8,
                                   scale=1.)
            vis_name = out_name + '_vis.png'
            _scipy.misc.imsave(vis_name, visim)
            if write_pmaps:
                VIS_FAC = 255.
                vismaps = _np.max(pmaps, axis=2)
                visim = _np.tile((vismaps * VIS_FAC).astype('uint8')[:, :, None],
                                 (1, 1, 3))
                if _CV2_AVAILABLE:
                    visim = _cv2.applyColorMap(visim, _cv2.COLORMAP_AUTUMN)[:, :, ::-1]
                    visim = (visim.astype('float32') * 0.6 +
                             image.astype('float32') * 0.4).astype('uint8')
                else:
                    visim = vismaps * 255
                vis_name = pmap_out_name + '_vis.png'
                _scipy.misc.imsave(vis_name, visim)



if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO, format=LOGFORMAT)
    # pylint: disable=no-value-for-parameter
    predict_pose_from()
