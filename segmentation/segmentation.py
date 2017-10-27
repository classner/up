#!/usr/bin/env python
"""
Segmentation predictions by the 31-part DLRN101 model.

Caffe must be available on the Pythonpath for this to work. The methods can
be imported and used directly, or the command line interface can be used. In
the latter case, adjust the log-level to your needs.

Author: Christoph Lassner.
"""
# pylint: disable=invalid-name
import os as _os
import logging as _logging
import glob as _glob
import numpy as _np
import scipy as _scipy
import click as _click
import sys  # pylint: disable=wrong-import-order
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from config import DEEPLAB_BUILD_FP
sys.path.insert(0, DEEPLAB_BUILD_FP)
import caffe as _caffe  # pylint: disable=import-error
import up_tools.model as mdl
try:
    import cv2 as _cv2  # pylint: disable=wrong-import-order
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_LOGGER = _logging.getLogger(__name__)

# Constants.
_MEAN = _np.array([104.008, 116.669, 122.675])  # In BGR order.

# CNN model store.
_MODEL = None

_BORDER = 0
_RF = 513
_MAXSIZE = 513
_ISIZE_STEP = 513
_DETECTION_THRESHOLD = 0.

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def segment_person(image, pose=None, probabilities=True):
    """
    Get the segmentation for an image.

    If pose is given, only segment the respective area. If pose is not given,
    the image must fit the receptive field size of the network, i.e., 513x513.

    If OpenCV is available, small, unconnected blobs will be removed.

    Parameters
    ==========

    :param image: np.array(3D).
      The image in height X width X BGR format.

    :param pose: np.array(2D) or None.
      The estimated pose in (2+) X {14; 91} format or None.

    Returns
    =======

    :param segmentation: np.array(3D).
      Depending on the parameter `probabilities`, returns
      a numpy array with height x width x 31 with the probabilities
      for the parts or the human, if `probabilities` is set to True.
    """
    global _MODEL  # pylint: disable=global-statement
    _LOGGER.info("Loading human part segmentation model...")
    _MODEL = _caffe.Net(
        _os.path.join(_os.path.abspath(_os.path.dirname(__file__)),
                      '..',
                      'models',
                      'segmentation',
                      'testpy_test_31_500_pkg_dorder.prototxt'),
        _os.path.join(_os.path.abspath(_os.path.dirname(__file__)),
                      '..',
                      'models',
                      'segmentation',
                      'test2.caffemodel'),
        _caffe.TEST)
    _LOGGER.info("Done!")
    _LOGGER.debug("Processing image...")
    model = _MODEL
    if pose is not None:
        pose = pose.copy()
        # Get an initial estimate of the bounding box.
        boxminy = max(0, _np.min(pose[1])-20)
        boxmaxy = min(_np.max(pose[1])+20, image.shape[0] - 1)
        boxminx = max(0, _np.min(pose[0]-20))
        boxmaxx = min(_np.max(pose[0])+20, image.shape[1] - 1)
        # Rescale and cut out.
        detsize = float(max(boxmaxx - boxminx, boxmaxy - boxminy))
        scale = 470. / detsize
        pose *= scale
        tim = _scipy.misc.imresize(image, scale, interp='bilinear')
        dsim = tim.copy()
        boxminy = int(max(0, _np.min(pose[1])-20))
        boxmaxy = int(min(_np.max(pose[1])+20, tim.shape[0] - 1))
        boxminx = int(max(0, _np.min(pose[0]-20)))
        boxmaxx = int(min(_np.max(pose[0])+20, tim.shape[1] - 1))
        tim = tim[boxminy:boxmaxy+1, boxminx:boxmaxx+1, :]
    else:
        tim = image
    # Process.
    _LOGGER.debug("Prepared image shape: %s.", str(tim.shape))
    assert _np.all(_np.array(tim.shape)[:2] <= _MAXSIZE), 'Input image shape too big!'
    prepim = (tim - _MEAN).transpose((2, 0, 1))
    results = _process_image(prepim, model)

    # Cleanup.
    out_c = 31
    if probabilities:
        upscale_prep = _np.zeros((dsim.shape[0],
                                  dsim.shape[1],
                                  out_c), dtype='float32')
        upscale_res = _np.empty((image.shape[0],
                                 image.shape[1],
                                 out_c), dtype='float32')
        upscale_prep[boxminy:boxmaxy+1, boxminx:boxmaxx+1, :] = \
                                                results.transpose((1, 2, 0))
        for c_idx in range(out_c):
            upscale_res[:, :, c_idx] = _scipy.misc.imresize(
                upscale_prep[:, :, c_idx],
                (image.shape[0],
                 image.shape[1]),
                interp='bilinear',
                mode='F')
    else:
        upscale_res = _np.zeros((dsim.shape[0],
                                 dsim.shape[1],
                                 3), dtype='uint8')
        upscale_res[boxminy:boxmaxy+1, boxminx:boxmaxx+1, :] =\
            _np.tile(_np.argmax(results, axis=0).astype('uint8')[:, :, None],
                     (1, 1, 3))
        upscale_res = _scipy.misc.imresize(upscale_res,
                                           (image.shape[0], image.shape[1]),
                                           interp='nearest')
    if not probabilities and _CV2_AVAILABLE and False:
        upscale_res = upscale_res[:, :, 0].copy()
        # Filter for small, unattached areas.
        contours, _ = _cv2.findContours(upscale_res.copy(),
                                        _cv2.RETR_CCOMP,
                                        _cv2.CHAIN_APPROX_SIMPLE)
        small_blobs = []
        # Find indices of contours whose area is less than `threshold`.
        for contour_idx, contour in enumerate(contours):
            contour_area = _cv2.contourArea(contour)
            if contour_area < 100:
                small_blobs.append(contour_idx)
        # Fill-in all small contours with zeros.
        for contour_idx in small_blobs:
            _cv2.drawContours(upscale_res,
                              contours,
                              contour_idx,
                              0,  # color
                              -1,  # thickness
                              8)  # lineType
        upscale_res = _np.tile(upscale_res[:, :, None], (1, 1, 3))
    _LOGGER.debug("Segmentation estimated.")
    return upscale_res


def _process_image(image, model):
    result = _np.zeros((model.blobs['prob'].shape[1],
                        image.shape[1],
                        image.shape[2]), dtype='float32')
    counter = _np.zeros((result.shape[1], result.shape[2]), dtype='float32')
    image_padded = _np.zeros((image.shape[0],
                              image.shape[1] + _BORDER * 2,
                              image.shape[2] + _BORDER * 2), dtype='float32')
    image_padded[:,
                 _BORDER:_BORDER+image.shape[1],
                 _BORDER:_BORDER+image.shape[2]] = image
    max_inp_size = max(image_padded.shape[1], image_padded.shape[2])
    target_inp_size = int(_np.ceil(float(min(max(max_inp_size, _RF), _MAXSIZE)) /
                                   float(_ISIZE_STEP)) * _ISIZE_STEP)
    _LOGGER.debug("Determined network input size: %d.", target_inp_size)
    model.blobs['data'].reshape(model.blobs['data'].shape[0],
                                3,
                                target_inp_size,
                                target_inp_size)
    # Copy in the patches.
    patch_offs = []
    #residx = 0
    for y_off in [0]:
        for x_off in [0]:
            # Account for the border overlap.
            _LOGGER.debug("Collecting patch for %d, %d...",
                          y_off, x_off)
            model.blobs['data'].data[len(patch_offs), ...] = 0.
            patch = image_padded[:,
                                 y_off:y_off+target_inp_size,
                                 x_off:x_off+target_inp_size]
            model.blobs['data'].data[len(patch_offs),
                                     :,
                                     :patch.shape[1],
                                     :patch.shape[2]] = patch
            patch_offs.append((y_off, x_off))
            if len(patch_offs) == model.blobs['data'].data.shape[0]:
                _LOGGER.debug("Forwarding through model...")
                model.forward()
                for patch_idx, (y_off, x_off) in enumerate(patch_offs):
                    res = result[:,
                                 y_off:y_off + target_inp_size - 2 * _BORDER,
                                 x_off:x_off + target_inp_size - 2 * _BORDER]
                    valid_patch = model.blobs['prob'].data[patch_idx,
                                                              :,
                                                              _BORDER:_BORDER+res.shape[1],
                                                              _BORDER:_BORDER+res.shape[2]]
                    _LOGGER.debug("Patch foreground probability sum: %f.",
                                  _np.sum(valid_patch[1:]))
                    if _np.sum(valid_patch[1:]) > _DETECTION_THRESHOLD:
                        res += valid_patch
                        counter[y_off:y_off + target_inp_size - 2 * _BORDER,
                                x_off:x_off + target_inp_size - 2 * _BORDER] += 1.
                        #_cv2.imwrite('out_%d-%f.png' % (residx, _np.sum(valid_patch[1:])),
                        #             model.blobs['prob'].data[patch_idx, 1] * 255.)
                        #_cv2.imwrite('res_%d.png' % residx, res[1, :, :] * 255.)
                        #residx += 1
                patch_offs = []
    if len(patch_offs) > 0:
        _LOGGER.debug("Forwarding through model...")
        model.forward()
        for patch_idx, (y_off, x_off) in enumerate(patch_offs):
            res = result[:,
                         y_off:y_off + target_inp_size - 2 * _BORDER,
                         x_off:x_off + target_inp_size - 2 * _BORDER]
            valid_patch = model.blobs['prob'].data[patch_idx,
                                                      :,
                                                      _BORDER:_BORDER+res.shape[1],
                                                      _BORDER:_BORDER+res.shape[2]]
            _LOGGER.debug("Patch foreground probability sum: %f.",
                          _np.sum(valid_patch[1:]))
            if _np.sum(valid_patch[1:]) > _DETECTION_THRESHOLD:
                res += valid_patch
                counter[y_off:y_off + target_inp_size - 2 * _BORDER,
                        x_off:x_off + target_inp_size - 2 * _BORDER] += 1.
                #_cv2.imwrite('out_%d.png' % residx,
                #             model.blobs['prob'].data[patch_idx, 1] * 255.)
                #_cv2.imwrite('res_%d.png' % residx, res[1, :, :] * 255.)
                #residx += 1
        patch_offs = []
    if result.max() > 0:
        result /= counter
    # Background for no detections.
    for ch_idx in range(result.shape[0]):
        channel = result[ch_idx]
        if ch_idx == 0:
            channel[counter == 0] = 1.
        else:
            channel[counter == 0] = 0.
    return result


###############################################################################
# Command line interface.
###############################################################################

@_click.command()
@_click.argument('image_name',
                 type=_click.Path(exists=True, dir_okay=True, readable=True))
@_click.option('--pose_name',
               type=_click.Path(exists=True, dir_okay=True, readable=True),
               help=('Specify a pose-containing .npz file. If not provided, '
                     'assumes `image_name`_pose.npz. If image_name is a '
                     'directory, poses must be stored per image accordingly in '
                     'a directory with `pose_name`.'),
               default=None)
@_click.option('--out_name',
               type=_click.Path(dir_okay=True, writable=True),
               help='Where to store the result.',
               default=None)
@_click.option('--probabilities',
               type=_click.BOOL,
               is_flag=True,
               help='Create a probability map instead of hard assignments.',
               default=False)
@_click.option('--visualize',
               type=_click.BOOL,
               help='Whether to visualize the result. Default: True.',
               default=True)
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help='The ending to use for the images to read, if a folder is specified.',
               default='.png')
@_click.option('--use_cpu',
               type=_click.BOOL,
               is_flag=True,
               help='Use CPU instead of GPU for predictions.',
               default=False)
@_click.option('--allow_subsampling',
               type=_click.BOOL,
               is_flag=True,
               help='Skip images without pose.',
               default=False)
# pylint: disable=too-many-arguments
def predict_segmentation_from(image_name,
                              pose_name=None,
                              out_name=None,
                              probabilities=False,
                              visualize=True,
                              folder_image_suffix='.png',
                              use_cpu=False,
                              allow_subsampling=False):
    """Load an image file, predict the segmentation and write it."""
    if _os.path.isdir(image_name):
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = _glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix))
        images = [im_fp for im_fp in images if not im_fp.endswith('vis.png')]
        process_folder = True
    else:
        images = [image_name]
        process_folder = False
    if use_cpu:
        _caffe.set_mode_cpu()
    else:
        _caffe.set_mode_gpu()
    out_name_provided = out_name
    pose_name_provided = pose_name
    if process_folder and out_name is not None and not _os.path.exists(out_name):
        _os.mkdir(out_name)
    for image_name in images:
        if out_name_provided is None:
            if probabilities:
                out_name = image_name + '_psegmentation_prob.npz'
            else:
                out_name = image_name + '_psegmentation.npz'
        elif process_folder:
            if probabilities:
                out_name = _os.path.join(out_name_provided,
                                         _os.path.basename(image_name) +
                                         '_psegmentation_prob.npz')
            else:
                if probabilities:
                    out_name = _os.path.join(out_name_provided,
                                             _os.path.basename(image_name) +
                                             '_segmentation_prob.npz')
                else:
                    out_name = _os.path.join(out_name_provided,
                                             _os.path.basename(image_name) +
                                             '_segmentation.npz')
        if pose_name_provided is None:
            pose_name = image_name + '_pose.npz'
        elif process_folder:
            pose_name = _os.path.join(pose_name_provided,
                                      _os.path.basename(image_name) + '_pose.npz')
        if not _os.path.exists(pose_name):
            _LOGGER.warning("No pose file found (%s)!", pose_name)
            pose = None
            if allow_subsampling:
                # Skip.
                continue
        else:
            pose = _np.load(pose_name)['pose']
        _LOGGER.info("Predicting the segmentation on `%s` (saving to `%s`) with "
                     "pose from `%s`).", image_name, out_name, pose_name)
        image = _scipy.misc.imread(image_name)
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, ::-1]

        segmentation = segment_person(image, pose, probabilities)
        _np.savez_compressed(out_name, segmentation=segmentation)
        if visualize:
            if probabilities:
                prep_segmentation = _np.argmax(segmentation, axis=2)
                VIS_FAC = 255. / 31.
                visim = _np.tile((prep_segmentation * VIS_FAC).astype('uint8')[:, :, None],
                                 (1, 1, 3))
            else:
                # Map back the colors.
                segmentation = segmentation[:, :, 0]
                seg_orig = _np.ones_like(image) * 255
                for y_idx in range(segmentation.shape[0]):
                    for x_idx in range(segmentation.shape[1]):
                        if segmentation[y_idx, x_idx] > 0:
                            seg_orig[y_idx, x_idx, :] = mdl.regions.reverse_mapping.keys()[  # pylint: disable=no-member
                                segmentation[y_idx, x_idx] - 1]  # pylint: disable=no-member
                # Save.
                #_scipy.misc.imsave(out_name + '_orig_vis.png', seg_orig)
                # Blend.
                #_scipy.misc.imsave(out_name + '_orig_blend_vis.png',
                #                   (seg_orig.astype('float32') * 0.5 +
                #                    image[:, :, ::-1].astype('float32') * 0.5).astype('uint8'))
                # Blend only on foreground.
                blend_fg = image[:, :, ::-1].copy()
                fg_regions = _np.dstack([(segmentation != 0)[:, :, None] for _ in range(3)])
                blend_fg[fg_regions] = ((seg_orig[fg_regions].astype('float32') * 0.5 +
                                         image[fg_regions].astype('float32') * 0.5).astype('uint8'))  # pylint: disable=line-too-long
                visim = blend_fg
            vis_name = out_name + '_vis.png'
            _scipy.misc.imsave(vis_name, visim)


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    # pylint: disable=no-value-for-parameter
    predict_segmentation_from()
