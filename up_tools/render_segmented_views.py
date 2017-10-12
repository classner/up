#!/usr/bin/env python
"""Data management script for this project."""
# pylint: disable=invalid-name, wrong-import-order
from __future__ import print_function

import sys
import logging as _logging
from copy import copy as _copy
from glob import glob as _glob
import cPickle as _pickle
import os as _os
import os.path as _path

import numpy as _np
import cv2 as _cv2
import click as _click
import pymp as _pymp

import opendr.renderer as _odr_r
import opendr.camera as _odr_c
import opendr.lighting as _odr_l

import tqdm
# pylint: disable=no-name-in-module
from up_tools.mesh import Mesh as _Mesh
from up_tools.bake_vertex_colors import bake_vertex_colors
from up_tools.camera import (rotateY as _rotateY)
from clustertools.log import LOGFORMAT
from clustertools.config import available_cpu_count

sys.path.insert(0, _path.join(_path.dirname(__file__),
                              '..', '..'))
from config import SMPL_FP
sys.path.insert(0, SMPL_FP)
try:
    from smpl.serialization import load_model  # pylint: disable=import-error
except ImportError:
    try:
        from psbody.smpl.serialization import load_model  # pylint: disable=import-error
    except ImportError:
        from smpl_webuser.serialization import load_model

_LOGGER = _logging.getLogger(__name__)
MODEL_NEUTRAL_PATH = _os.path.join(
    _path.dirname(__file__), '..', 'models', '3D',
    'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
MODEL_NEUTRAL = load_model(MODEL_NEUTRAL_PATH)
_TEMPLATE_MESH = _Mesh(filename=_os.path.join(_os.path.dirname(__file__),
                                              '..', 'models', '3D',
                                              'template-bodyparts.ply'))

def _rodrigues_from_seq(angles_seq):
    """Create rodrigues representation of angles."""
    rot = _np.eye(3)
    for angle in angles_seq[::-1]:
        rot = rot.dot(_cv2.Rodrigues(angle)[0])
    return _cv2.Rodrigues(rot)[0].flatten()


def _create_renderer(  # pylint: disable=too-many-arguments
        w=640,
        h=480,
        rt=_np.zeros(3),
        t=_np.zeros(3),
        f=None,
        c=None,
        k=None,
        near=1.,
        far=10.,
        texture=None):
    """Create a renderer for the specified parameters."""
    f = _np.array([w, w]) / 2. if f is None else f
    c = _np.array([w, h]) / 2. if c is None else c
    k = _np.zeros(5)           if k is None else k

    if texture is not None:
        rn = _odr_r.TexturedRenderer()
    else:
        rn = _odr_r.ColoredRenderer()  # pylint: disable=redefined-variable-type

    rn.camera = _odr_c.ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near':near, 'far':far, 'height':h, 'width':w}
    if texture is not None:
        rn.texture_image = _np.asarray(_cv2.imread(texture), _np.float64)/255.
    return rn


def _stack_with(rn, mesh, texture):
    if texture is not None:
        if not hasattr(mesh, 'ft'):
            mesh.ft = mesh.f
            mesh.vt = mesh.v[:, :2]
        rn.ft = _np.vstack((rn.ft, mesh.ft+len(rn.vt)))
        rn.vt = _np.vstack((rn.vt, mesh.vt))
    rn.f = _np.vstack((rn.f, mesh.f+len(rn.v)))
    rn.v = _np.vstack((rn.v, mesh.v))
    rn.vc = _np.vstack((rn.vc, mesh.vc))


def _simple_renderer(rn, meshes, yrot=0, texture=None, use_light=False):
    mesh = meshes[0]
    if texture is not None:
        if not hasattr(mesh, 'ft'):
            mesh.ft = _copy(mesh.f)
            vt = _copy(mesh.v[:, :2])
            vt -= _np.min(vt, axis=0).reshape((1, -1))
            vt /= _np.max(vt, axis=0).reshape((1, -1))
            mesh.vt = vt
        mesh.texture_filepath = rn.texture_image

    # Set camera parameters
    if texture is not None:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, ft=mesh.ft, vt=mesh.vt, bgcolor=_np.ones(3))
    else:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, bgcolor=_np.ones(3))

    for next_mesh in meshes[1:]:
        _stack_with(rn, next_mesh, texture)

    # Construct light.
    if use_light:
        albedo = rn.vc
        rn.vc = _odr_l.LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=_rotateY(_np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=_np.array([1, 1, 1]))
        # Construct Left Light
        rn.vc += _odr_l.LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=_rotateY(_np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=_np.array([1, 1, 1]))

        # Construct Right Light
        rn.vc += _odr_l.LambertianPointLight(
            f=rn.f,
            v=rn.v,
            num_verts=len(rn.v),
            light_pos=_rotateY(_np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=_np.array([.7, .7, .7]))
    return rn.r


# pylint: disable=too-many-locals
def render(model, resolution, cam, steps, segmented=False, use_light=False):  # pylint: disable=too-many-arguments
    """Render a sequence of views from a fitted body model."""
    assert steps >= 1
    if segmented:
        texture = _os.path.join(_os.path.dirname(__file__),
                                '..', 'models', '3D', 'mask_filled.png')
    else:
        texture = _os.path.join(_os.path.dirname(__file__),
                                '..', 'models', '3D', 'mask_filled_uniform.png')
    mesh = _copy(_TEMPLATE_MESH)

    # render ply
    model.betas[:len(cam['betas'])] = cam['betas']
    model.pose[:] = cam['pose']
    model.trans[:] = cam['trans']

    mesh.v = model.r
    w, h = resolution[0], resolution[1]
    dist = _np.abs(cam['t'][2] - _np.mean(mesh.v, axis=0)[2])
    rn = _create_renderer(w=w,
                          h=h,
                          near=1.,
                          far=20.+dist,
                          rt=_np.array(cam['rt']),
                          t=_np.array(cam['t']),
                          f=_np.array([cam['f'], cam['f']]),
                          # c=_np.array(cam['cam_c']),
                          texture=texture)
    light_yrot = _np.radians(120)
    baked_mesh = bake_vertex_colors(mesh)
    base_mesh = _copy(baked_mesh)
    mesh.f = base_mesh.f
    mesh.vc = base_mesh.vc
    renderings = []
    for angle in _np.linspace(0., 2. * (1. - 1. / steps) * _np.pi, steps):
        mesh.v = _rotateY(base_mesh.v, angle)
        imtmp = _simple_renderer(rn=rn,
                                 meshes=[mesh],
                                 yrot=light_yrot,
                                 texture=texture,
                                 use_light=use_light)
        im = _np.zeros(h*w*3).reshape(((h, w, 3)))
        im[:h, :w, :] = imtmp*255.
        renderings.append(im)
    return renderings

###############################################################################
# Command-line interface.
###############################################################################


@_click.command()
@_click.argument('input_folder', type=_click.Path(exists=True, file_okay=False,
                                                  readable=True))
@_click.argument('out_folder', type=_click.Path(writable=True))
@_click.option("--num_shots_per_body", type=_click.INT, default=7,
               help="Number of shots to take per body.")
@_click.option("--only_missing", type=_click.BOOL, default=False, is_flag=True,
               help="Only run for missing shots.")
@_click.option("--num_threads", type=_click.INT, default=-1,
               help="Number of threads to use.")
@_click.option("--use_light", type=_click.BOOL, default=False, is_flag=True,
               help="Light the scene for a depth effect.")
@_click.option("--factor", type=_click.FLOAT, default=1.,
               help="Scaling factor for the rendered human (not the image).")
def sample_shots(  # pylint: disable=too-many-arguments
        input_folder,
        out_folder,
        num_shots_per_body=7,
        only_missing=False,
        num_threads=-1,
        use_light=False,
        factor=1.):
    """Sample body images with visibilities."""
    _LOGGER.info("Sampling 3D body shots.")
    if num_threads == -1:
        num_threads = available_cpu_count()
    else:
        assert num_threads > 0
    if not _path.exists(out_folder):
        _os.mkdir(out_folder)
    _np.random.seed(1)
    bodies = _glob(_path.join(input_folder, '*.pkl'))
    _LOGGER.info("%d bodies detected.", len(bodies))
    with _pymp.Parallel(num_threads, if_=num_threads > 1) as p:
        for body_idx in p.iterate(tqdm.tqdm(range(len(bodies)))):
            body_filename = bodies[body_idx]
            vis_filename = body_filename + '_vis_overlay.png'
            vis_filename = body_filename + '_vis.png'
            if not _os.path.exists(vis_filename):
                vis_filename = body_filename + '_vis_0.png'
            if not _os.path.exists(vis_filename):
                # Try something else.
                vis_filename = body_filename[:-9]
            out_names = [_os.path.join(out_folder,
                                       _path.basename(body_filename) + '.' +
                                       str(map_idx) + '.png')
                         for map_idx in range(num_shots_per_body)]
            if only_missing:
                all_exist = True
                for fname in out_names:
                    if not _path.exists(fname):
                        all_exist = False
                        break
                if all_exist:
                    continue
            vis_im = _cv2.imread(vis_filename)
            assert vis_im is not None, 'visualization not found: %s' % (vis_filename)
            renderings = render_body_impl(body_filename,
                                          [vis_im.shape[1], vis_im.shape[0]],
                                          num_shots_per_body,
                                          quiet=False,
                                          use_light=use_light,
                                          factor=factor)
            for map_idx, vmap in enumerate(renderings):
                _cv2.imwrite(out_names[map_idx], vmap[:, :, ::-1])
    _LOGGER.info("Done.")


def render_body_impl(filename,  # pylint: disable=too-many-arguments
                     resolution=None,
                     num_steps_around_y=1,
                     quiet=False,
                     use_light=False,
                     factor=1.):
    """Create a SMPL rendering."""
    if resolution is None:
        resolution = [640, 480]
    if not quiet:
        _LOGGER.info("Rendering SMPL model from `%s` in resolution %s.",
                     filename, str(resolution))
        _LOGGER.info("    Using %d steps around y axis.",
                     num_steps_around_y)
    with open(filename, 'rb') as inf:
        camera = _pickle.load(inf)
    # Render.
    renderings = render(
        MODEL_NEUTRAL,
        (_np.asarray(resolution) * 1. / factor).astype('int'),
        camera,
        num_steps_around_y,
        False,
        use_light=use_light)
    interp = 'bilinear' if use_light else 'nearest'
    import scipy.misc
    renderings = [scipy.misc.imresize(renderim,
                                      (resolution[1],
                                       resolution[0]),
                                      interp=interp)
                  for renderim in renderings]
    if not quiet:
        _LOGGER.info("Done.")
    return renderings


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO,
                         format=LOGFORMAT,)
                         # filename=__file__+'_log.txt')
    _logging.getLogger("opendr.lighting").setLevel(_logging.WARN)
    sample_shots()  # pylint: disable=no-value-for-parameter
