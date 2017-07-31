"""Visualization tools."""
# pylint: disable=no-member, invalid-name
import clustertools.visualization as vs
from up_tools.model import (connections_lsp,
                            connections_landmarks_91,
                            lm_region_mapping)


def visualize_pose(image,  # pylint: disable=too-many-arguments, dangerous-default-value
                   pose,
                   line_thickness=3,
                   dash_length=15,
                   opacity=0.6,
                   circle_color=(255, 255, 255),
                   connections=[],
                   region_mapping=None,
                   skip_unconnected_joints=True,
                   scale=1.):
    """Visualize a pose."""
    if pose.shape[1] == 91:
        this_connections = connections
        if this_connections == []:
            this_connections = connections_landmarks_91
        this_lm_region_mapping = region_mapping
        if this_lm_region_mapping is None:
            this_lm_region_mapping = lm_region_mapping
    else:
        this_connections = connections
        if this_connections == []:
            this_connections = connections_lsp
        this_lm_region_mapping = region_mapping
    return vs.visualize_pose(image, pose, line_thickness, dash_length,
                             opacity, circle_color, this_connections,
                             this_lm_region_mapping,
                             skip_unconnected_joints,
                             scale)
