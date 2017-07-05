#!/usr/bin/env python2
"""Get evaluation results for stored scoremaps."""
# pylint: disable=invalid-name
from __future__ import print_function
from os import path
import logging

import numpy as np
from PIL import Image
import scipy.ndimage
import click
import tqdm
import pymp
from clustertools.config import available_cpu_count  # pylint: disable=unused-import
from clustertools.log import LOGFORMAT

from up_tools.model import regions, six_region_groups



LOGGER = logging.getLogger(__name__)

# pylint: disable=no-member
VOC_REGION_GROUPS = [
    [regions.ruhead, regions.rlhead, regions.luhead, regions.llhead],
    [regions.neck, regions.rubody, regions.rlbody, regions.lubody,
     regions.llbody, regions.rshoulder, regions.lshoulder],
    # Lower arms.
    [regions.rlarm, regions.rwrist, regions.rhand,
     regions.llarm, regions.lwrist, regions.lhand],
    [regions.ruarm, regions.luarm, regions.relbow, regions.lelbow],
    # Lower legs.
    [regions.rlleg, regions.rankle, regions.rfoot,
     regions.llleg, regions.lankle, regions.lfoot],
    [regions.ruleg, regions.luleg, regions.rknee, regions.lknee],
]


# pylint: enable=no-member
@click.command()
@click.argument("image_list_file", type=click.Path(dir_okay=False, readable=True))
@click.argument("data_folder", type=click.Path(file_okay=False, readable=True))
@click.argument("result_label_folder", type=click.Path(file_okay=False, readable=True))
@click.argument("n_labels", type=click.INT)
@click.option("--as_nolr", type=click.BOOL, is_flag=True, default=False,
              help="Evaluate 6 class body part segmentation without lr.")
@click.option("--ev_31_as_6", type=click.BOOL, is_flag=True, default=False,
              help="Evaluate 31 region predictions as 6 regions (n_labels must be 6).")
def main(image_list_file,  # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
         data_folder,
         result_label_folder,
         n_labels,
         as_nolr=False,
         ev_31_as_6=False):
    """Perform the evaluation for previously written results scoremaps."""
    LOGGER.info("Evaluating segmentation in folder `%s`.", result_label_folder)
    if 'voc' in image_list_file:
        voc_mode = True
        ev_31_as_6 = True
        n_labels = 7
        LOGGER.info("Using VOC part segmentation style.")
    else:
        voc_mode = False
    if ev_31_as_6:
        assert n_labels == 7
    if as_nolr:
        assert n_labels == 7
    classIOUs = np.zeros((n_labels,))
    overallIOU = 0.
    overallAccuracy = 0.
    # Parallel stats.
    TP = pymp.shared.array((n_labels,), dtype='float32')
    FP = pymp.shared.array((n_labels,), dtype='float32')
    FN = pymp.shared.array((n_labels,), dtype='float32')
    imgTP = pymp.shared.array((1,), dtype='float32')
    imgPixels = pymp.shared.array((1,), dtype='float32')
    stat_lock = pymp.shared.lock()
    warned = False
    with open(image_list_file, 'r') as inf:
        image_list = inf.readlines()
    with pymp.Parallel(available_cpu_count(), if_=False and available_cpu_count() > 1) as p:
        for imgnames in p.iterate(tqdm.tqdm(image_list), element_timeout=20):  # pylint: disable=too-many-nested-blocks
            imgname = imgnames.split(" ")[0].strip()[1:]
            old_applied_scale = float(imgnames.split(" ")[2].strip())
            gtname = imgnames.split(" ")[1].strip()[1:]
            gt_file = path.join(data_folder, gtname)
            gtLabels = np.array(Image.open(gt_file))
            if gtLabels.ndim == 3:
                if not warned:
                    LOGGER.warn("Three-layer ground truth detected. Using first.")
                    warned = True
                gtLabels = gtLabels[:, :, 0]
            gtLabels = scipy.misc.imresize(gtLabels, 1. / old_applied_scale,
                                           interp='nearest')
            if as_nolr:
                gtLabels[gtLabels == 4] = 3
                gtLabels[gtLabels == 6] = 5
            LOGGER.debug("Evaluating `%s`...", imgname)
            result_file = path.join(result_label_folder,
                                    path.basename(imgname) + '.npy')
            result_probs = np.load(result_file)
            result_probs = np.array(
                [scipy.misc.imresize(layer, 1. / old_applied_scale, interp='bilinear', mode='F')
                 for layer in result_probs])
            if result_probs.shape[0] > n_labels and not (
                    ev_31_as_6 and result_probs.shape[0] == 32):
                LOGGER.warn('Result has invalid labels: %s!',
                            str(result_probs.shape[0]))
                continue
            if result_probs.min() < 0 or result_probs.max() > 1.:
                LOGGER.warn('Invalid result probabilities: min `%f`, max `%f`!',
                            result_probs.min(), result_probs.max())
                continue
            else:
                MAP = np.argmax(result_probs, axis=0)
                if as_nolr:
                    MAP[MAP == 4] = 3
                    MAP[MAP == 6] = 5
                if ev_31_as_6:
                    if voc_mode:
                        groups_to_use = VOC_REGION_GROUPS
                    else:
                        groups_to_use = six_region_groups
                    for classID in range(1, 32):
                        new_id = -1
                        for group_idx, group in enumerate(groups_to_use):
                            for grelem in group:
                                if regions.reverse_mapping.keys().index(grelem) == classID - 1:  # pylint: disable=no-member
                                    new_id = group_idx + 1
                        assert new_id > 0
                        if not voc_mode:
                            gtLabels[gtLabels == classID] = new_id
                        MAP[MAP == classID] = new_id
                for classID in range(n_labels):
                    classGT = np.equal(gtLabels, classID)
                    classResult = np.equal(MAP, classID)
                    classResult[np.equal(gtLabels, 255)] = 0
                    with stat_lock:
                        TP[classID] = TP[classID] + np.count_nonzero(classGT & classResult)
                        FP[classID] = FP[classID] + np.count_nonzero(classResult & ~classGT)
                        FN[classID] = FN[classID] + np.count_nonzero(~classResult & classGT)
                imgResult = MAP
                imgGT = gtLabels
                imgResult[np.equal(MAP, 255)] = 0
                imgGT[np.equal(gtLabels, 255)] = 0
                with stat_lock:
                    imgTP[0] += np.count_nonzero(np.equal(imgGT, imgResult))
                    imgPixels[0] += np.size(imgGT)
    for classID in range(0, n_labels):
        classIOUs[classID] = TP[classID] / (TP[classID] + FP[classID] + FN[classID])
    if as_nolr:
        classIOUs = classIOUs[(0, 1, 2, 3, 5),]
    overallIOU = np.mean(classIOUs)
    overallAccuracy = imgTP[0] / imgPixels[0]
    if n_labels == 32:
        region_names = ['background'] + regions.reverse_mapping.values()[:-1]  # pylint: disable=no-member
        LOGGER.info("Class IOUs:")
        for region_name, class_iou in zip(region_names, classIOUs):
            LOGGER.info('%s: %f', region_name, class_iou)
    else:
        LOGGER.info('Class IOUs: %s.', str(classIOUs))
    LOGGER.info('Overall IOU: %s.', str(overallIOU))
    LOGGER.info('Overall Accuracy: %s.', str(overallAccuracy))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    main()  # pylint: disable=no-value-for-parameter
