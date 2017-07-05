#!/usr/bin/env python2
"""Write the segmentation results to disk for further evaluation."""
from os import path
import sys
import logging
from PIL import Image
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))

import numpy as np
import click
import tqdm

from clustertools.visualization import apply_colormap
from clustertools.log import LOGFORMAT

from config import DEEPLAB_BUILD_FP  # pylint: disable=import-error

LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("caffe_prototxt", type=click.Path(dir_okay=False, readable=True))
@click.argument("caffe_model", type=click.Path(dir_okay=False, readable=True))
@click.argument("image_folder", type=click.Path(file_okay=False, readable=True))
@click.argument("image_list_file", type=click.Path(dir_okay=False, readable=True))
@click.argument("output_folder", type=click.Path(file_okay=False, writable=True))
@click.option("--caffe_install_path",
              type=click.Path(file_okay=False, readable=True),
              default=path.join(DEEPLAB_BUILD_FP, 'install/bin/python'))
@click.option("--n_labels", type=click.INT, default=31)
def main(caffe_prototxt,  # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
         caffe_model,
         image_folder,
         image_list_file,
         output_folder,
         caffe_install_path,
         n_labels):
    """Store and visualize the segmentation results for a model."""
    LOGGER.info("Storing segmentation results to folder `%s`.",
                output_folder)
    LOGGER.info("Using caffe from `%s`.", caffe_install_path)
    sys.path.insert(0, path.join(caffe_install_path))
    import caffe  # pylint: disable=import-error
    mean_red = 122.675
    mean_green = 116.669
    mean_blue = 104.008

    # Configure preprocessing
    caffe.set_mode_gpu()
    net_full_conv = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
    net_input_blob = net_full_conv.inputs[0]
    transformer = caffe.io.Transformer({
        net_full_conv.inputs[0]: net_full_conv.blobs[net_full_conv.inputs[0]].data.shape})
    transformer.set_transpose(net_input_blob, (2, 0, 1))
    transformer.set_channel_swap(net_input_blob, (2, 1, 0))
    transformer.set_raw_scale(net_input_blob, 255.0)
    net_inp_height, net_inp_width = net_full_conv.blobs[net_input_blob].data.shape[2:4]
    # Create and configure the mean image. The transformer applies channel-swap
    # first, so we have BGR order for the mean image.
    mean_image = np.zeros((3, net_inp_height, net_inp_width), dtype='float32')
    mean_image[0, :, :] = mean_blue
    mean_image[1, :, :] = mean_green
    mean_image[2, :, :] = mean_red
    transformer.set_mean(net_input_blob, mean_image)
    with open(image_list_file, 'r') as inf:
        image_list = inf.readlines()
    for imgnames in tqdm.tqdm(image_list):
        imgname = imgnames.split(" ")[0][1:].strip()
        LOGGER.debug("Processing `%s`...", imgname)
        image_filename = path.join(image_folder, imgname)
        # caffe.io loads as RGB, and in range [0., 1.].
        im = caffe.io.load_image(image_filename)  # pylint: disable=invalid-name
        height, width = im.shape[:2]
        # Pad values.
        pad_width = net_inp_width - width
        pad_height = net_inp_height - height
        im = np.lib.pad(im,  # pylint: disable=invalid-name
                        ((0, pad_height),
                         (0, pad_width),
                         (0, 0)),
                        'constant',
                        constant_values=-5)
        assert im.shape[0] == net_inp_height
        assert im.shape[1] == net_inp_width
        R = im[:, :, 0]  # pylint: disable=invalid-name
        G = im[:, :, 1]  # pylint: disable=invalid-name
        B = im[:, :, 2]  # pylint: disable=invalid-name
        # Will be multiplied by 255 by the transformer.
        R[R == -5] = mean_red / 255.
        G[G == -5] = mean_green / 255.
        B[B == -5] = mean_blue / 255.
        im[:, :, 0] = R
        im[:, :, 1] = G
        im[:, :, 2] = B
        out = net_full_conv.forward_all(
            data=np.asarray([transformer.preprocess(net_input_blob, im)]))
        pmap = out['prob'][0]
        assert pmap.min() >= 0. and pmap.max() <= 1., (
            "Invalid probability value in result map!")
        prob_map = pmap[:, :height, :width]
        np.save(path.join(output_folder,
                          path.basename(imgname) + '.npy'),
                prob_map)
        maxed_map = np.argmax(prob_map, axis=0)
        vis_image = Image.fromarray(apply_colormap(maxed_map, vmax=n_labels-1))
        vis_image.save(path.join(output_folder,
                                 path.basename(imgname) + '.npy.vis.png'))
        raw_image = Image.fromarray(maxed_map.astype('uint8'))
        raw_image.save(path.join(output_folder,
                                 path.basename(imgname) + '_segmentation.png'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    main()  # pylint: disable=no-value-for-parameter
