#!/usr/bin/env python2
"""Create a rendering from a segmentation in the original 31 colors."""
import os.path as path
import logging
from glob import glob
from scipy.misc import imread, imsave
import numpy as np
import click
import up_tools.model as mdl
from clustertools.log import LOGFORMAT
import tqdm


LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument('seg_fp', type=click.Path(exists=True, readable=True,
                                          writable=True, file_okay=False))
@click.option('--image_fp', type=click.Path(exists=True, readable=True),
              help='Look for images here.', default=None)
def cli(seg_fp, image_fp=None):
    """Create a rendering from a segmentation in the original 31 colors."""
    if image_fp is None:
        image_fp = seg_fp
    LOGGER.info("Applying original colormap on segmentations in folder `%s`.",
                seg_fp)
    for segim_fp in tqdm.tqdm(glob(path.join(seg_fp, '*_segmentation.png'))):
        seg = imread(segim_fp)
        im = imread(path.join(image_fp,  # pylint: disable=invalid-name
                              path.basename(segim_fp)[:-len('_segmentation.png')]))
        assert np.all(seg.shape[:2] == im.shape[:2])
        # Map back the colors.
        seg_orig = np.ones_like(im) * 255
        for y_idx in range(seg.shape[0]):
            for x_idx in range(seg.shape[1]):
                if seg[y_idx, x_idx] > 0:
                    seg_orig[y_idx, x_idx, :] = mdl.regions.reverse_mapping.keys()[  # pylint: disable=no-member
                        seg[y_idx, x_idx] - 1]
        # Save.
        imsave(segim_fp + '_orig_vis.png', seg_orig)
        # Blend.
        imsave(segim_fp + '_orig_blend_vis.png',
               (seg_orig.astype('float32') * 0.5 +
                im.astype('float32') * 0.5).astype('uint8'))
        # Blend only on foreground.
        blend_fg = im.copy()
        fg_regions = np.dstack([(seg != 0)[:, :, None] for _ in range(3)])
        blend_fg[fg_regions] = ((seg_orig[fg_regions].astype('float32') * 0.5 +
                                 im[fg_regions].astype('float32') * 0.5).astype('uint8'))
        imsave(segim_fp + '_orig_blend_fg_vis.png', blend_fg)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()  # pylint: disable=no-value-for-parameter
