#!/usr/bin/env python2
"""Bake the vertex colors by superposition."""
from copy import copy
import numpy as np
import click

import opendr.renderer as _odr_r
import opendr.camera as _odr_c
from up_tools.mesh import Mesh


def bake_vertex_colors(inmesh):
    """Bake the vertex colors by superposition."""
    faces = np.arange(inmesh.f.size).reshape(-1, 3)
    vertices = np.empty((len(inmesh.f)*3, 3))
    vc = np.zeros_like(vertices)  # pylint: disable=invalid-name
    tmpar = np.ascontiguousarray(inmesh.vc).view(
        np.dtype((np.void, inmesh.vc.dtype.itemsize * inmesh.vc.shape[1])))
    _, unique_idx = np.unique(tmpar, return_index=True)
    unique_clrs = inmesh.vc[unique_idx]
    for iface, face in enumerate(inmesh.f):
        vertices[iface*3+0] = inmesh.v[face[0]]
        vertices[iface*3+1] = inmesh.v[face[1]]
        vertices[iface*3+2] = inmesh.v[face[2]]
        low_idx = np.argmin([np.linalg.norm(inmesh.vc[face[0]]),
                             np.linalg.norm(inmesh.vc[face[1]]),
                             np.linalg.norm(inmesh.vc[face[2]])])
        vc[iface*3+0] = inmesh.vc[face[low_idx]]
        vc[iface*3+1] = inmesh.vc[face[low_idx]]
        vc[iface*3+2] = inmesh.vc[face[low_idx]]
    tmpar = np.ascontiguousarray(vc).view(np.dtype((np.void, vc.dtype.itemsize * vc.shape[1])))
    _, unique_idx = np.unique(tmpar, return_index=True)
    unique_clrs_after = vc[unique_idx]
    for clr in unique_clrs:
        assert clr in unique_clrs_after
    for clr in unique_clrs_after:
        assert clr in unique_clrs
    outmesh = Mesh(v=vertices, f=faces, vc=vc)
    return outmesh


def get_face_for_pixel(shape, mesh, model, configuration, coords):  # pylint: disable=too-many-locals
    """Get the face index or -1 for the mesh in the given conf at coords."""
    assert len(shape) == 2, str(shape)
    assert np.all(coords >= 0), str(coords)
    assert coords.ndim == 2, str(coords.ndim)
    for coord in coords.T:
        assert coord[0] < shape[1], "%s, %s" % (str(coord), str(shape))
        assert coord[1] < shape[0], "%s, %s" % (str(coord), str(shape))
    mesh = copy(mesh)
    # Setup the model.
    model.betas[:len(configuration['betas'])] = configuration['betas']
    model.pose[:] = configuration['pose']
    model.trans[:] = configuration['trans']
    mesh.v = model.r
    inmesh = mesh
    # Assign a different color for each face.
    faces = np.arange(inmesh.f.size).reshape(-1, 3)
    vertices = np.empty((len(inmesh.f)*3, 3))
    vc = np.zeros_like(vertices)  # pylint: disable=invalid-name
    for iface, face in enumerate(inmesh.f):
        vertices[iface*3+0] = inmesh.v[face[0]]
        vertices[iface*3+1] = inmesh.v[face[1]]
        vertices[iface*3+2] = inmesh.v[face[2]]
        vc[iface*3+0] = (float(iface % 255) / 255., float(iface / 255) / 255., 0.)
        vc[iface*3+1] = (float(iface % 255) / 255., float(iface / 255) / 255., 0.)
        vc[iface*3+2] = (float(iface % 255) / 255., float(iface / 255) / 255., 0.)
    fcmesh = Mesh(v=vertices, f=faces, vc=vc)
    # Render the mesh.
    dist = np.abs(configuration['t'][2] - np.mean(fcmesh.v, axis=0)[2])
    rn = _odr_r.ColoredRenderer()  # pylint: disable=redefined-variable-type, invalid-name
    rn.camera = _odr_c.ProjectPoints(
        rt=configuration['rt'],
        t=configuration['t'],
        f=np.array([configuration['f'], configuration['f']]),
        c=np.array([shape[1], shape[0]]) / 2.,
        k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': dist + 20., 'height': shape[0], 'width': shape[1]}
    rn.set(v=fcmesh.v, f=fcmesh.f, vc=fcmesh.vc, bgcolor=np.ones(3))
    rendered = rn.r
    results = [-1 for _ in range(len(coords.T))]
    for coord_idx, coord in enumerate(coords.T):
        # Find the face or background.
        loc_color = (rendered[int(coord[1]), int(coord[0])] * 255.).astype('uint8')
        if np.all(loc_color == 255):
            continue
        else:
            assert loc_color[2] == 0, str(loc_color)
            face_idx = loc_color[1] * 255 + loc_color[0]
            assert face_idx >= 0 and face_idx < len(mesh.f)
            results[coord_idx] = face_idx
    return results


@click.command()
@click.argument("inmesh_fp", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("outmesh_fp", type=click.Path(dir_okay=False, writable=True))
def cli(inmesh_fp, outmesh_fp):
    """Bake the vertex colors by superposition."""
    inmesh = Mesh(filename=inmesh_fp)
    outmesh = bake_vertex_colors(inmesh)
    outmesh.write_ply(outmesh_fp)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
