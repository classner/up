#!/usr/bin/env python2
"""Configuration values for the project."""
import os.path as path
import click


############################ EDIT HERE ########################################
SMPL_FP = path.expanduser("~/smpl")
DEEPLAB_BUILD_FP = path.expanduser("~/git/deeplab-public-ver2/build")
DEEPERCUT_CNN_BUILD_FP = path.expanduser("~/git/deepcut-cnn/build_cluster")
UP3D_FP = path.expanduser("~/datasets/up-3d")
SEG_DATA_FP = path.expanduser("~/datasets/seg_prepared")
POSE_DATA_FP = path.expanduser("~/datasets/pose_prepared")
DIRECT3D_DATA_FP = path.expanduser("~/datasets/2dto3d_prepared")


###############################################################################
# Infrastructure. Don't edit.                                                 #
###############################################################################

@click.command()
@click.argument('key', type=click.STRING)
def cli(key):
    """Print a config value to STDOUT."""
    if key in globals().keys():
        print globals()[key]
    else:
        raise Exception("Requested configuration value not available! "
                        "Available keys: " +
                        str([kval for kval in globals().keys() if kval.isupper()]) +
                        ".")


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter

