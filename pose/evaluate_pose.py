#!/usr/bin/env python2
"""Get evaluation results for stored landmarks."""
# pylint: disable=invalid-name, wrong-import-order
from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from os import path
import logging

import numpy as np
import click

from clustertools.config import available_cpu_count  # pylint: disable=unused-import


LOGGER = logging.getLogger(__name__)


def _getDistPCK(pred, gt, norm_lm_ids):
    """
    Calculate the pck distance for all given poses.

    norm_lm_ids: Use the distance of these landmarks for normalization. Usually
                 lshoulder and rhip.
    """
    assert pred.ndim == gt.ndim
    assert pred.ndim == 3
    assert pred.shape[0] >= 2
    assert np.all(pred.shape[1:] == gt.shape[1:])
    dist = np.empty((1, pred.shape[1], pred.shape[2]))

    for imgidx in range(pred.shape[2]):
        # torso diameter
        refDist = np.linalg.norm(gt[:2, norm_lm_ids[0], imgidx] -
                                 gt[:2, norm_lm_ids[1], imgidx])
        # distance to gt joints
        dist[0, :, imgidx] =\
            np.sqrt(
                np.sum(
                    np.power(pred[:2, :, imgidx] -
                             gt[:2, :, imgidx], 2),
                    axis=0)) / refDist
    return dist

def _computePCK(dist, rnge, mask):
    """Compute PCK values for given joint distances and a range."""
    pck = np.zeros((len(rnge), dist.shape[1] + 1))
    for joint_idx in range(dist.shape[1]):
        # compute PCK for each threshold
        evaluation_basis = dist[0,
                                joint_idx,
                                np.where(mask[joint_idx, :] > 0)]
        for k, rngval in enumerate(rnge):
            pck[k, joint_idx] = 100. *\
                np.mean(evaluation_basis.flat <= rngval)
    # compute average PCK
    for k in range(len(rnge)):
        pck[k, -1] = np.mean(pck[k, :-1])
    return pck

def _area_under_curve(xpts, ypts):
    """Calculate the AUC."""
    a = np.min(xpts)
    b = np.max(xpts)
    # remove duplicate points
    _, I = np.unique(xpts, return_index=True)  # pylint: disable=W0632
    xpts = xpts[I]
    ypts = ypts[I]
    assert np.all(np.diff(xpts) > 0)
    if len(xpts) < 2:
        return np.NAN
    from scipy import integrate
    myfun = lambda x: np.interp(x, xpts, ypts)
    auc = integrate.quad(myfun, a, b)[0]
    return auc

def PCK(poses,  # pylint: disable=too-many-arguments
        annotations,
        norm_lm_ids,
        plot=False,
        rnge_max=0.2,
        print_res=False,
        using_joint_index=-1):
    r"""
    Implementation of the PCK measure.

    As defined in Sapp&Taskar, CVPR 2013.
    Torso height: ||left_shoulder - right hip||.
    Validated to give the same results as Pishchulin et al.

    Parameters
    ==========

    :param poses: np.ndarray((M>2, L, N)).
      M are the coordinates, L joints, N is
      the number of samples.

    :param annotations: np.ndarray((O>2, L, N)).
      The annotated poses. The coordinate order must match the pose coordinate
      order.

    :param norm_lm_ids: 2-tuple(int).
      The indices of the two landmarks to use for normalization.

    :param plot: bool.
      Whether to directly show a plot of the results.

    :param rnge_max: float.
      Up to which point to calculate the AUC.

    :param print_res: bool.
      Whether to print a summary of results.

    :param using_joint_index: int.
      If > 1, specifies a column in the pose array, which indicates binary
      whether to take the given joint into account or not.
    """
    assert using_joint_index == -1 or using_joint_index > 1
    assert len(norm_lm_ids) == 2
    rnge = np.arange(0., rnge_max + 0.001, 0.01)
    dist = _getDistPCK(poses, annotations, norm_lm_ids)
    # compute PCK
    if using_joint_index > 1:
        mask = poses[using_joint_index, :, :] > 0
    else:
        mask = np.ones((poses.shape[1], poses.shape[2]))
    pck = _computePCK(dist, rnge, mask)
    auc = _area_under_curve(rnge / rnge.max(), pck[:, -1])
    if plot:
        plt.plot(rnge,
                 pck[:, -1],
                 label='PCK',
                 linewidth=2)
        plt.xlim(0., 0.2)
        plt.xticks(np.arange(0, rnge_max + 0.01, 0.02))
        plt.yticks(np.arange(0, 101., 10.))
        plt.ylabel('Detection rate, %')
        plt.xlabel('Normalized distance')
        plt.grid()
        legend = plt.legend(loc=4)
        legend.get_frame().set_facecolor('white')
        plt.show()
    # plot(range,pck(:,end),'color',p.colorName,
    #'LineStyle','-','LineWidth',3);
    if print_res:
        # pylint: disable=superfluous-parens
        print("AUC: {}.".format(auc))
        print("@0.2: {}.".format(pck[np.argmax(rnge > 0.2) - 1, -1]))
    return rnge, pck, auc


@click.command()    
@click.argument("image_list_file", type=click.Path(dir_okay=False, readable=True))
@click.argument("scale_fp", type=click.Path(dir_okay=False, readable=True))
@click.argument("result_label_folder", type=click.Path(file_okay=False, readable=True))
@click.argument("n_labels", type=click.INT)
def main(image_list_file,  # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
         scale_fp,
         result_label_folder,
         n_labels):
    """Perform the evaluation for previously written result landmarks."""
    LOGGER.info("Evaluating landmarks in folder `%s`.", result_label_folder)
    LOGGER.info("Reading image information...")
    with open(image_list_file, 'r') as inf:
        image_list_lines = inf.readlines()
    with open(scale_fp, 'r') as inf:
        scale_lines = inf.readlines()
    all_scales = dict((line.split(" ")[0].strip(), float(line.split(" ")[1].strip()))
                      for line in scale_lines)
    lm_annots = OrderedDict()
    read_lms = None
    for line in image_list_lines:
        if line.startswith('#'):
            if read_lms is not None:
                lm_annots[imname] = read_lms[:]  # pylint: disable=used-before-assignment, unsubscriptable-object
            image_started = True
            read_lms = []
        elif image_started:
            imname = line.strip()
            image_started = False
            size_spec = 0
        elif size_spec < 3:
            size_spec += 1
            continue
        elif size_spec == 3:
            num_lms = int(line.strip())
            assert num_lms == n_labels
            size_spec += 1
        else:
            read_lms.append((int(line.split(" ")[1].strip()),
                             int(line.split(" ")[2].strip())))
    scales = [all_scales[path.basename(imname)] for imname in lm_annots.keys()]
    annots = np.array(lm_annots.values()).transpose((2, 1, 0)).astype('float')
    for annot_idx in range(annots.shape[2]):
        annots[:2, :, annot_idx] /= scales[annot_idx]
    LOGGER.info("Loading results...")
    lm_positions = []
    for imgname, scale in zip(lm_annots.keys(), scales):
        result_file = path.join(result_label_folder,
                                path.basename(imgname) + '.npy')
        lm_positions.append(np.load(result_file) / scale)
    lm_positions = np.array(lm_positions).transpose((1, 2, 0))
    LOGGER.info("Evaluating...")
    if lm_positions.shape[1] == 91:
        from model import landmarks_91
        rnge, pck, auc = PCK(lm_positions, annots,
                             [landmarks_91.lshoulder,  # pylint: disable=no-member
                              landmarks_91.rhip],  # pylint: disable=no-member
                             print_res=False,
                             plot=False)
    else:
        # Assume LSP model.
        rnge, pck, auc = PCK(lm_positions, annots,
                             (9, 2),
                             print_res=False,
                             plot=False)
    # Create the plot.
    plt.figure(figsize=(7, 7))
    plt.plot(rnge,
             pck[:, -1],
             label='PCK',
             linewidth=2)
    plt.xlim(0., 0.2)
    plt.xticks(np.arange(0, 0.2 + 0.01, 0.02))
    plt.yticks(np.arange(0, 101., 10.))
    plt.ylabel('Detection rate, %')
    plt.xlabel('Normalized distance')
    plt.grid()
    legend = plt.legend(loc=4)
    legend.get_frame().set_facecolor('white')
    plt.savefig(path.join(result_label_folder, 'pck.png'))
    # plot(range,pck(:,end),'color',p.colorName,
    #'LineStyle','-','LineWidth',3);
    LOGGER.info("AUC: %f.", auc)
    LOGGER.info("@0.2: %f.", pck[np.argmax(rnge > 0.2) - 1, -1])
    LOGGER.info("Per-part information (PCK@0.2):")
    if lm_positions.shape[1] == 91:
        for lmid, lmname in landmarks_91.reverse_mapping.items():  # pylint: disable=no-member
            LOGGER.info("%s %f", lmname, pck[np.argmax(rnge > 0.2) - 1, lmid])
    else:
        from PoseKit.model import joints_lsp
        valsatp2 = []
        for lmid, lmname in joints_lsp.reverse_mapping.items():  # pylint: disable=no-member
            LOGGER.info("%s %f", lmname, pck[np.argmax(rnge > 0.2) - 1, lmid])
            valsatp2.append(pck[np.argmax(rnge > 0.2) - 1, lmid])
        LOGGER.info("PCK@0.2 wo neck and head: %f.", np.mean(valsatp2[:-2]))
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
