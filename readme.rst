Description
===========

AWarpSharp2 sharpens edges by warping them. This is a rewrite of the
Avisynth filter (re)written by SEt.

This fork uses the VapourSynth API 4 (VapourSynth R55 or newer) and
builds wheels for Windows x64, Linux x86_64 and macOS arm64 (see
`Fork changes`_). x86 uses SSE2 (``opt=True``), other CPUs
(e.g. arm64) the C code, which gives the same results.

The plugin registers the namespace ``warp`` with four functions:
AWarpSharp2, ASobel, ABlur and AWarp.


Usage
=====
::

    warp.AWarpSharp2(clip clip[, int thresh=128, int blur=2, int type=0, int[] depth=[16, 8, 8], int chroma=0, int[] planes=<all>, bint opt=True, string cplace="mpeg1"])

    warp.ASobel(clip clip[, int thresh=128, int[] planes=<all>, bint opt=True])

    warp.ABlur(clip clip[, int blur=3, int type=1, int[] planes=<all>, bint opt=True])

    warp.AWarp(clip clip, clip mask[, int[] depth=[3, 1, 1], int chroma=0, int[] planes=<all>, bint opt=True, string cplace="mpeg1"])

AWarpSharp2 performs edge detection, blurring, and warping, all in one.

ASobel performs edge detection, with an algorithm that might resemble the
one used by std.Sobel.

ABlur blurs.

AWarp warps.

Parameters:
    *clip*
        Clip to process. Must have constant format and dimensions,
        8..16 bit integer pixels, and it must not be RGB. Gray and YUV
        with chroma subsampling up to 2 (4:2:0, 4:2:2, 4:4:0, 4:4:4)
        are accepted.

        For the AWarp filter, *clip* must be either the same size as
        *mask*, or four times the size of *mask* in each dimension. The
        latter can be useful if better subpixel interpolation is desired,
        e.g. using nnedi3_rpow2.

        If *clip* is upscaled to four times the original size, it must be
        top-left aligned. See the examples below.

    *mask*
        Edge mask. Must have the same format as *clip*. The output has
        the dimensions and the length of *mask*; if *clip* is shorter,
        its last frame is repeated.

    *thresh*
        No pixel in the edge mask will have a value greater than *thresh*.
        Decrease for weaker sharpening.

        Range: 0..255. It will be scaled internally for bit depths greater
        than 8.

        Default: 128.

    *blur*
        Controls the number of times to blur the edge mask. Increase for
        weaker sharpening.

        Range: 0..+infinity.

        Default: 2 for *type* 0, 3 for *type* 1.

    *type*
        Controls the type of blur to use. 0 means some kind of 13x13
        average. 1 means some kind of 5x5 average. 1 is better, but
        requires more passes for similar sharpening strength as 0.

        Default: 0.

    *depth*
        Controls how far to warp. Negative values warp in the other
        direction, i.e. will blur the image instead of sharpening.

        Range: -128..127

        Default for depth[0]: 16 for AWarpSharp2, 3 for AWarp.
        Default for depth[1]: depth[0] // 2.
        Default for depth[2]: depth[1].

    *chroma*
        Controls the chroma handling method. 0 will use the edge mask from
        the luma to warp the chroma. 1 will create an edge mask from each
        chroma channel and use those to warp each chroma channel
        individually.

        Default: 0.

    *planes*
        Controls which planes are processed. Planes that are not processed
        are simply copied from the source frame, except in the AWarp filter,
        when *clip* is four times the size of *mask*, in which case the
        unprocessed planes will contain uninitialised memory.

        Default: all.

    *opt*
        If True, the best functions supported by your CPU will be used.
        If False, plain C++ functions will be used. On CPUs other than
        x86 the C++ functions are always used.

        Default: True.

    *cplace*
        Chroma placement. Must be either "mpeg1" or "mpeg2".

        Only relevant when *chroma* is 0 and the input is 4:2:0 or
        4:2:2.

        Default: "mpeg1".


Examples
========

::

    import vapoursynth as vs

    core = vs.core

    src = core.d2v.Source("d2v.d2v")

    mask = core.warp.ASobel(src)
    mask = core.warp.ABlur(mask)

    # Use sx=0.375 sy=0.375 for top-left alignment.
    src4 = core.fmtc.resample(src, w=src.width * 4, h=src.height * 4, sx=0.375, sy=0.375)
    src4 = core.fmtc.bitdepth(src4, bits=8)

    warp = core.warp.AWarp(src4, mask)

The all-in-one filter with the same defaults::

    sharp = core.warp.AWarpSharp2(src)


Installation
============

Wheels for Windows x64, Linux x86_64 and macOS arm64 are attached to the
GitHub releases. They install the plugin into
``site-packages/vapoursynth/plugins``, where the VapourSynth Python
package autoloads it::

    pip install vapoursynth_awarpsharp2-<version>-py3-none-<platform>.whl

The plugin file inside the wheel (``awarpsharp2.dll``,
``libawarpsharp2.so`` or ``libawarpsharp2.dylib``) can also be copied
into any other VapourSynth plugin directory.


Compilation
===========

::

    meson setup build
    ninja -C build

Without VapourSynth development files on the system (pkg-config
``vapoursynth``), the vendored API 4 headers in ``include/vapoursynth``
are used.

Wheel (installs into ``site-packages/vapoursynth/plugins``)::

    pip install .

The GitHub Actions workflow in ``.github/workflows`` builds and tests
the wheels for all three platforms and attaches them to a release when a
``v*`` tag is pushed.

Tests (the plugin must be autoloaded, or pass its path)::

    python test/test_awarpsharp2.py [build/awarpsharp2.dll]


Fork changes
============

- VapourSynth API 4 (``VapourSynthPluginInit2``), plugin version 4.1;
  VapourSynth R55 or newer is required.
- Builds and runs on arm64 (macOS): the C++ code is used there, with the
  same results as the SSE2 code on x86 (checked by the tests for every
  function in 8, 10 and 16 bit).
- The Meson build of the original enabled the SSE2 code under a
  misspelled define, so ``opt=True`` had no effect in Meson builds;
  the SSE2 code is now used on x86.
- AWarp: when *clip* is shorter than *mask*, the clip's last frame is
  repeated instead of requesting a frame beyond its end.
- AWarp: an invalid *planes* argument no longer leaks the *mask* node.
- Everywhere else the output is identical to the original plugin.
- Meson build with vendored API 4 headers and meson-python wheels for
  Windows x64, Linux x86_64 and macOS arm64; the autotools build was
  removed.


License
=======

ISC.
