"""
gtstyle.color -- sRGB <-> CIELAB conversion (D65 white point, sRGB
companding) and the Lab-space colormap interpolation used by the Figure 2 and
Figure 3 panels, which build a colormap by interpolating a short list of RGB
anchors through CIELAB space rather than RGB space and converting back.

This module implements the conversions directly from the CIE formulas rather
than delegating to skimage.color, which is not a dependency of this package.
tests/test_color.py checks them against reference values:
    white [1,1,1]  -> L*a*b* = (100,   0,      0    )
    black [0,0,0]  -> L*a*b* = (0,     0,      0    )
    red   [1,0,0]  -> L*a*b* = (53.24, 80.09,  67.20)
"""

import numpy as np

# D65 reference white
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883

# sRGB <-> linear-light XYZ, D65 (IEC 61966-2-1)
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)

_DELTA = 6.0 / 29.0


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c, dtype=float)
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _f(t):
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _finv(t):
    return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def gt_rgb2lab(rgb):
    """Convert sRGB in [0, 1] to CIELAB under the D65 white point.

    rgb: (...,3) array. Returns an array of the same leading shape with L in
    [0, 100] and a, b roughly in [-100, 100].
    """
    rgb = np.asarray(rgb, dtype=float)
    lin = _srgb_to_linear(rgb)
    xyz = lin @ _M_RGB2XYZ.T

    fx = _f(xyz[..., 0] / _XN)
    fy = _f(xyz[..., 1] / _YN)
    fz = _f(xyz[..., 2] / _ZN)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def gt_lab2rgb(lab):
    """Convert CIELAB under the D65 white point to sRGB.

    lab: (...,3) array. Returns an array of the same leading shape. Values
    outside the sRGB gamut are returned unclipped; callers that require
    [0, 1] clip afterwards (gt_lab_colormap does so).
    """
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    X = _XN * _finv(fx)
    Y = _YN * _finv(fy)
    Z = _ZN * _finv(fz)

    xyz = np.stack([X, Y, Z], axis=-1)
    lin = xyz @ _M_XYZ2RGB.T
    return _linear_to_srgb(lin)


def gt_lab_colormap(cm_input, n):
    """Build an n-step colormap by interpolating RGB anchors through CIELAB.

    The anchors are converted to CIELAB, linearly interpolated to n evenly
    spaced steps in that space, and converted back to sRGB. Interpolating in
    CIELAB rather than RGB keeps the perceived lightness gradient even.

    cm_input: (p,3) array of RGB anchors in [0, 1]. n: number of steps.
    Returns an (n,3) array clipped to [0, 1].
    """
    cm_input = np.asarray(cm_input, dtype=float)
    p = cm_input.shape[0]
    lab = gt_rgb2lab(cm_input)

    x_old = np.arange(1, p + 1)
    x_new = np.linspace(1, p, n)
    lab_i = np.column_stack([
        np.interp(x_new, x_old, lab[:, 0]),
        np.interp(x_new, x_old, lab[:, 1]),
        np.interp(x_new, x_old, lab[:, 2]),
    ])
    cm = gt_lab2rgb(lab_i)
    return np.clip(cm, 0.0, 1.0)
