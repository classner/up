#!/usr/bin/env python
"""Render a model."""
import cPickle as pickle
import logging as _logging
import os as _os
import os.path as _path
# pylint: disable=invalid-name
import sys as _sys
from config import SMPL_FP
from copy import copy

import click as _click
import cv2
import numpy as np
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer, TexturedRenderer

from up_tools.camera import rotateY
from up_tools.mesh import Mesh

_sys.path.insert(0, _path.join(_path.dirname(__file__), '..'))
_sys.path.insert(0, SMPL_FP)
try:
    from smpl.serialization import load_model as _load_model
except:
    from smpl_webuser.serialization import load_model as _load_model

_LOGGER = _logging.getLogger(__name__)
_TEMPLATE_MESH = Mesh(filename=_os.path.join(_os.path.dirname(__file__),
                                             '..', 'models', '3D', 'template.ply'))
_TEMPLATE_MESH_SEGMENTED = Mesh(filename=_os.path.join(_os.path.dirname(__file__),
                                                       '..', 'models', '3D', 'template-bodyparts.ply'))
_COLORS = {
    'pink': [.6, .6, .8],
    'cyan': [.7, .75, .5],
    'yellow': [.5, .7, .75],
    'grey': [.7, .7, .7],
}


def _create_renderer(  # pylint: disable=too-many-arguments
        w=640,
        h=480,
        rt=np.zeros(3),
        t=np.zeros(3),
        f=None,
        c=None,
        k=None,
        near=1.,
        far=10.,
        texture=None):
    """Create a renderer for the specified parameters."""
    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5)           if k is None else k

    if texture is not None:
        rn = TexturedRenderer()
    else:
        rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near':near, 'far':far, 'height':h, 'width':w}
    if texture is not None:
        rn.texture_image = np.asarray(cv2.imread(texture), np.float64)/255.
    return rn


def _stack_with(rn, mesh, texture):
    if texture is not None:
        if not hasattr(mesh, 'ft'):
            mesh.ft = mesh.f
            mesh.vt = mesh.v[:, :2]
        rn.ft = np.vstack((rn.ft, mesh.ft+len(rn.vt)))
        rn.vt = np.vstack((rn.vt, mesh.vt))
    rn.f = np.vstack((rn.f, mesh.f+len(rn.v)))
    rn.v = np.vstack((rn.v, mesh.v))
    rn.vc = np.vstack((rn.vc, mesh.vc))


def _simple_renderer(rn, meshes, yrot=0, texture=None):
    mesh = meshes[0]
    if texture is not None:
        if not hasattr(mesh, 'ft'):
            mesh.ft = copy(mesh.f)
            vt = copy(mesh.v[:, :2])
            vt -= np.min(vt, axis=0).reshape((1, -1))
            vt /= np.max(vt, axis=0).reshape((1, -1))
            mesh.vt = vt
        mesh.texture_filepath = rn.texture_image

    # Set camers parameters
    if texture is not None:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, ft=mesh.ft, vt=mesh.vt, bgcolor=np.ones(3))
    else:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, bgcolor=np.ones(3))

    for next_mesh in meshes[1:]:
        _stack_with(rn, next_mesh, texture)

    # Construct Back Light (on back right corner)
    albedo = rn.vc

    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


# pylint: disable=too-many-locals
def render(model, image, cam, steps, segmented=False, scale=1.):
    """Render a sequence of views from a fitted body model."""
    assert steps >= 1
    if segmented:
        texture = None
        mesh = copy(_TEMPLATE_MESH_SEGMENTED)
    else:
        texture = None
        mesh = copy(_TEMPLATE_MESH)
        mesh.vc = _COLORS['pink']
    #cmesh = Mesh(_os.path.join(_os.path.dirname(__file__),
    #                           'template-bodyparts-corrected-labeled-split5.ply'))
    #mesh.vc = cmesh.vc.copy()
    # render ply
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']
    model.trans[:] = cam['trans']

    mesh.v = model.r
    w, h = (image.shape[1], image.shape[0])
    dist = np.abs(cam['t'][2] - np.mean(mesh.v, axis=0)[2])
    rn = _create_renderer(w=int(w * scale),
                          h=int(h * scale),
                          near=4.,
                          far=30.+dist,
                          rt=np.array(cam['rt']),
                          t=np.array(cam['t']),
                          f=np.array([cam['f'], cam['f']]) * scale,
                          # c=np.array(cam['cam_c']),
                          texture=texture)
    light_yrot = np.radians(120)
    base_mesh = copy(mesh)
    renderings = []
    for angle in np.linspace(0., 2. * np.pi, num=steps, endpoint=False):
        mesh.v = rotateY(base_mesh.v, angle)
        imtmp = _simple_renderer(rn=rn,
                                 meshes=[mesh],
                                 yrot=light_yrot,
                                 texture=texture)
        if segmented:
            imtmp = imtmp[:, :, ::-1]
        renderings.append(imtmp * 255.)
    return renderings


@_click.command()
@_click.argument('filename', type=_click.Path(exists=True, readable=True))
@_click.option('--segmented',
               type=_click.BOOL,
               is_flag=True,
               help='If set, use a segmented mesh.')
@_click.option('--steps',
               type=_click.INT,
               help='The number of rotated images to render. Default: 1.',
               default=1)
@_click.option("--scale",
               type=_click.FLOAT,
               help="Render the results at this scale.",
               default=1.)
def cli(filename, segmented=False, steps=1, scale=1.):
    """Render a 3D model for an estimated body fit. Provide the image (!!) filename."""
    model = {
        'neutral': _load_model(
            _os.path.join(_os.path.dirname(__file__),
                          '..', 'models', '3D',
                          'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'))
    }
    fileending = filename[-4:]
    pkl_fname = filename.replace(fileending, '.pkl')
    if not _path.exists(pkl_fname):
        pkl_fname = filename[:-10] + '_body.pkl'
    if not _path.exists(pkl_fname):
        pkl_fname = filename + '_body.pkl'
    _LOGGER.info("Rendering 3D model for image %s and parameters %s...",
                 filename, pkl_fname)
    assert _os.path.exists(pkl_fname), (
        'Stored body fit does not exist for {}: {}!'.format(
            filename, pkl_fname))

    image = cv2.imread(filename)
    with open(pkl_fname) as f:
        cam = pickle.load(f)
    renderings = render(model['neutral'], image, cam, steps, segmented, scale)
    for ridx, rim in enumerate(renderings):
        if segmented:
            out_fname = filename + '_body_segmented_%d.png' % (ridx)
        else:
            out_fname = filename + '_body_%d.png' % (ridx)
        cv2.imwrite(out_fname, rim)


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    _logging.getLogger("opendr.lighting").setLevel(_logging.FATAL)  # pylint: disable=no-attribute
    cli()  # pylint: disable=no-value-for-parameter
