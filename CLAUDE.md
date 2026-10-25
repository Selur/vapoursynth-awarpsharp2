# Notes for Claude

## Project

VapourSynth plugin `warp` (id `com.nodame.awarpsharp2`) with four functions: AWarpSharp2
(edge mask + blur + warp in one), ASobel (edge mask), ABlur (mask blur), AWarp (warp a clip
along a given mask; the clip may be four times the mask's size). 8..16 bit integer, Gray or YUV
with chroma subsampling up to 2, not RGB. Fork of dubhater's plugin, ported to API 4
(VapourSynth R55+, `VapourSynthPluginInit2`). User documentation is `readme.rst`.

- `src/awarpsharp2.cpp`: C++ template kernels (`sobel_c`, `blur_r6_c`, `blur_r2_c`,
  `bilinear_downscale_*_c`, `warp_c<SMAGL, PixelType>`) followed by the API 4 glue (one
  `AWarpSharp2Data` struct for all four filters, `selectFunctions()` picks C, SSE2 or NEON kernels).
- `src/sse2.cpp`: SSE2 kernels (sobel/blur for 8 and 16 bit, warp for 8 bit only). Compiled only
  on x86 (`AWARPSHARP2_X86`, set by `meson.build`).
- `src/neon.cpp`: the same kernels for AArch64 (`AWARPSHARP2_ARM`), a lane-by-lane port of the
  SSE2 code (`Vec<PixelType>` traits for the 8/16 bit vector ops; the warp gathers its source
  pixels with scalar loads). Same loop bounds and edge handling as the SSE2 code, so the two can
  be compared line by line; a fix in one belongs in the other.
- `src/warp_edge.h`: scalar warp of one pixel (`warp_edge_c`), used by both SIMD files for the
  first and last pixel of every row.
- `opt=False` or any other CPU uses the C kernels; all three give bit-identical results. The SIMD
  kernels assume plane widths of at least 16 pixels (8 bit) like upstream; the pixel arithmetic is
  unchanged from upstream, only the position of the last vector per row was fixed (it started
  inside the left edge for narrow planes, and the warp wrote one byte past the row).
- Algorithm quirks that are upstream behaviour, not bugs: the last row of a warped plane is always
  interpolated with the row above (vertical limit for the last row is -1), so `depth=0` is not an
  exact identity there; AWarp with a 4x clip leaves unprocessed planes uninitialised.
- `include/vapoursynth/`: vendored R55 API 4 headers, used when pkg-config finds no `vapoursynth`.
- `test/test_awarpsharp2.py`: functional tests (SHA-256 references for every function in 8, 10
  and 16 bit, including 36 and 44 pixel wide clips whose chroma hits the last-vector cases; each
  case must agree between `opt=True` and `opt=False`; argument errors).
  Run with `python test/test_awarpsharp2.py build/awarpsharp2.dll`, or without the path when the
  wheel is installed. Changing pixel output changes the digests: regenerate them deliberately
  (set an entry to `None`, the test prints the new value).

## Build

- `meson setup build && ninja -C build`; on Windows with MSVC `meson setup build --vsenv -Db_vscrt=mt`.
- Wheel: `pip install .` (meson-python, `-Dpython_package=true` installs into
  `site-packages/vapoursynth/plugins`, where the VapourSynth PyPI package autoloads it).
- `.github/workflows/build-wheels.yml` builds and tests wheels for Windows x64 (MSVC, static
  runtime), Linux x86_64 (manylinux_2_28) and macOS arm64 with cibuildwheel, re-tags them
  `py3-none`, and attaches them to a GitHub release for `v*` tags. Only CI builds Linux and macOS;
  locally only the Windows build is checked.
- Checking SIMD against C without VapourSynth: a small harness that `#include`s `awarpsharp2.cpp`
  (the C kernels are `static` there) and calls the `*_neon`/`*_sse2` functions on random planes
  of many widths, heights and strides, comparing the pixels inside the plane. On an arm64 Mac the
  SSE2 kernels can be built with `clang++ -arch x86_64` and run under Rosetta.

## Versioning

Version numbers follow semantic versioning strictly. Never bump a version on
your own initiative:

- no bump for CI, packaging or build-system changes
- patch for bug fixes
- minor for backwards-compatible features
- major only for breaking changes

If a change seems to warrant a bump, propose it and let the maintainer decide.

## Format

- Always three parts, `MAJOR.MINOR.PATCH`, and the same number everywhere: the git tag
  (`vMAJOR.MINOR.PATCH`), `version` in `meson.build` (and with it the wheel name) and the
  release title.
- Where the code registers a VapourSynth plugin, `VS_MAKE_VERSION(major, minor)` in
  `configPlugin` carries MAJOR and MINOR; a patch release leaves it unchanged.
- Published releases are never renamed, retagged or rebuilt: their download URLs and SHA-256
  sums are pinned by Hybrid's build scripts. Older tags with one or two parts (`v5`, `v1.1`)
  stay as published; the next release continues from the current version with three parts,
  e.g. `5` -> `5.0.1`, `1.1` -> `1.1.1`.
