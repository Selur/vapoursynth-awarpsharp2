#!/usr/bin/env python3
"""Functional tests for the AWarpSharp2 VapourSynth plugin (API 4).

Usage: python test/test_awarpsharp2.py [path/to/awarpsharp2 plugin]

Without an argument the plugin is expected to be autoloaded (e.g. from the installed wheel).
"""
import ctypes
import hashlib
import sys

import vapoursynth as vs

core = vs.core
FAILURES = []

# SHA-256 of all planes of frame 0 for each case below (pattern_clip(100, 60) unless the case
# says otherwise). Every case runs with opt=True (SSE2 on x86, NEON on arm64) and opt=False
# (C code, the only code on other CPUs) and both must give exactly this result.
REFERENCE = {
    "AWarpSharp2()": "fe24eb7fb17d231a0d1597f18ee3380eabb64c848847a1ac3dd2918b8910add0",
    "AWarpSharp2(thresh=200, blur=3, type=1, depth=[24, 12, 6], chroma=1)": "9444fe011a0d19d7be25fe95d83d588c0cfab5d4658376f3e0c39e55c0278b64",
    "AWarpSharp2(cplace='mpeg2', depth=[-16, -8])": "bca0c416ddf4428ff054c849b8e1275fd1111224242b2f39d96acfa8d94101f6",
    "AWarpSharp2() 64x48": "d5a149addb86f0c62c948c840d07e330bc005ac1ceceb35392cdd27d26f065d6",
    "AWarpSharp2() YUV420P16": "bdc3e9e72a66de4a4f91d2ef4bf1aa94fa09cfc6d090833c85d237a63fa02b92",
    "AWarpSharp2(depth=[32, 16, 16]) YUV420P10": "94b77cfcd30bb8f7c6e16db0a5ae99b8b0bd4e146c80f91142ebe58f91f98636",
    "AWarpSharp2() YUV444P8": "8a535b9a9eeba40fc5524328a07f4f36b240152f44fcc1616195ce31dfb864fa",
    "AWarpSharp2(cplace='mpeg2') YUV422P8": "2f98f92a848dd515a3bad23a114ba5b770cc34376c2eafe4824bcc317c9e820b",
    "AWarpSharp2() GRAY8": "99c959c31bb2f74789f61426c20e80ab0b769652243b0f24cc578335348f487e",
    "ASobel(thresh=200)": "b3b76a940e8a285b173ee399cd1823c1df11fcb3f17b7e2c89fb771bac5e1b34",
    "ASobel() YUV420P16": "daa02d71e8147467334f44360f171ac4307e302bc0aa3c98bff3f0e1cd8d6706",
    "ABlur()": "95ff43fa222926e322d2eb920b0ab74717305f54f55e01469901362ad3a459fb",
    "ABlur(blur=2, type=0)": "4ba304f1c64272dde548ab00f85cc7634d1a10a49e1fdb66bd172f1058f2ce7d",
    "ABlur(blur=4, type=0) YUV420P16": "613fa25b733e5c0adc0ebdec03560799a46763edaa18c67c59dec71bfd4d4554",
    "AWarp(depth=[8, 4, 4])": "51f83129ac795fffd4dca9f362df05c80982204ee5df75053b262474d596cb82",
    "AWarp(depth=[8, 4, 4], chroma=1)": "7e0a2bd21d3c421e191b9d64bf76a8621f77896c3aecb595a62459b71dbb69fa",
    "AWarp() 4x clip": "2250f667fcffd34678642fe59fc1413f0ba61e4d3ad2b574ba0fd3cdd6bf6765",
    "AWarp() 4x clip YUV420P16": "55eb0f518512cda1dd64b22db9c1b96f227877793bb159b8ce1ae0d3701e6475",
    # Narrow chroma planes (18 and 22 pixels wide, processed directly with chroma=1 or by ABlur):
    # the last vector of the horizontal blur must not start inside the left edge.
    "AWarpSharp2(chroma=1) 36x28": "2ad9183dac80c528aade8e8a4898bed43c5b7e14c42b67de1efeed25cdcb7050",
    "AWarpSharp2(thresh=255, chroma=1) 44x28": "c18fd86d0d9c93a7358d79d8a7f2ae3139ce85f77a8569c69ec81c8bac3adb82",
    "ABlur(type=1) 36x28": "147b3b8814b3b1bb6121a05f86723ba31d3c0c2f49aa05cff863130651d46fb0",
    "ABlur(blur=1, type=0) 36x28 YUV420P16": "f213cf8821b025b8f808ad6bb839cfbe59e28b9606f99686c4135b842abcc817",
}


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("FAIL: " + msg)
        return
    print("ok  ", msg)


def expect_error(fn, msg, contains):
    try:
        fn()
    except vs.Error as e:
        check(contains in str(e), "%s (%s)" % (msg, str(e).strip().splitlines()[-1]))
    else:
        check(False, "no error raised: " + msg)


def pattern(w, h, n, p):
    """Gradient plus deterministic noise, with a few sharp edges."""
    rows = []
    state = (w * 7919 + h * 104729 + n * 1299709 + p * 15485863) & 0xFFFFFFFF
    for y in range(h):
        row = bytearray(w)
        for x in range(w):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            noise = (state >> 16) % 31 - 15
            base = 128 + ((x * 3 + y * 2 + n * 17 + p * 40) % 120) - 60
            if (x // 12 + y // 10) % 3 == 0:
                base += 70
            row[x] = max(0, min(255, base + noise))
        rows.append(bytes(row))
    return rows


def pattern_clip(w, h, fmt=vs.YUV420P8, length=1):
    """8 bit pattern clip; other integer formats are made from it with a bit shift (resize.Point)."""
    blank = core.std.BlankClip(width=w, height=h, format=vs.YUV420P8, length=length)

    def fill(n, f):
        fout = f.copy()
        for p in range(3):
            pw, ph = (w, h) if p == 0 else (w // 2, h // 2)
            ptr = fout.get_write_ptr(p)
            ptr = ptr.value if hasattr(ptr, "value") else ptr
            stride = fout.get_stride(p)
            for y, row in enumerate(pattern(pw, ph, n, p)):
                ctypes.memmove(ptr + y * stride, row, pw)
        return fout

    clip = core.std.ModifyFrame(blank, blank, fill)
    if fmt != vs.YUV420P8:
        clip = core.resize.Point(clip, format=fmt)
    return clip


def plane(clip, p, n=0):
    return bytes(clip.get_frame(n)[p])


def digest(clip, n=0):
    f = clip.get_frame(n)
    h = hashlib.sha256()
    for p in range(f.format.num_planes):
        h.update(bytes(f[p]))
    return h.hexdigest()


def cases():
    """name -> function(opt) returning the clip to hash."""
    src = pattern_clip(100, 60)
    src16 = pattern_clip(100, 60, vs.YUV420P16)
    src10 = pattern_clip(100, 60, vs.YUV420P10)
    src444 = pattern_clip(100, 60, vs.YUV444P8)
    src422 = pattern_clip(100, 60, vs.YUV422P8)
    gray = core.std.ShufflePlanes(src, 0, vs.GRAY)
    small = pattern_clip(64, 48)
    narrow36 = pattern_clip(36, 28)
    narrow44 = pattern_clip(44, 28)
    narrow36_16 = pattern_clip(36, 28, vs.YUV420P16)
    src4 = core.resize.Point(src, 400, 240)
    src4_16 = core.resize.Point(src16, 400, 240)

    def mask(clip, opt):
        return core.warp.ABlur(core.warp.ASobel(clip, opt=opt), opt=opt)

    return {
        "AWarpSharp2()": lambda opt: core.warp.AWarpSharp2(src, opt=opt),
        "AWarpSharp2(thresh=200, blur=3, type=1, depth=[24, 12, 6], chroma=1)":
            lambda opt: core.warp.AWarpSharp2(src, thresh=200, blur=3, type=1, depth=[24, 12, 6], chroma=1, opt=opt),
        "AWarpSharp2(cplace='mpeg2', depth=[-16, -8])":
            lambda opt: core.warp.AWarpSharp2(src, cplace="mpeg2", depth=[-16, -8], opt=opt),
        "AWarpSharp2() 64x48": lambda opt: core.warp.AWarpSharp2(small, opt=opt),
        "AWarpSharp2() YUV420P16": lambda opt: core.warp.AWarpSharp2(src16, opt=opt),
        "AWarpSharp2(depth=[32, 16, 16]) YUV420P10": lambda opt: core.warp.AWarpSharp2(src10, depth=[32, 16, 16], opt=opt),
        "AWarpSharp2() YUV444P8": lambda opt: core.warp.AWarpSharp2(src444, opt=opt),
        "AWarpSharp2(cplace='mpeg2') YUV422P8": lambda opt: core.warp.AWarpSharp2(src422, cplace="mpeg2", opt=opt),
        "AWarpSharp2() GRAY8": lambda opt: core.warp.AWarpSharp2(gray, opt=opt),
        "ASobel(thresh=200)": lambda opt: core.warp.ASobel(src, thresh=200, opt=opt),
        "ASobel() YUV420P16": lambda opt: core.warp.ASobel(src16, opt=opt),
        "ABlur()": lambda opt: core.warp.ABlur(src, opt=opt),
        "ABlur(blur=2, type=0)": lambda opt: core.warp.ABlur(src, blur=2, type=0, opt=opt),
        "ABlur(blur=4, type=0) YUV420P16": lambda opt: core.warp.ABlur(src16, blur=4, type=0, opt=opt),
        "AWarp(depth=[8, 4, 4])": lambda opt: core.warp.AWarp(src, mask(src, opt), depth=[8, 4, 4], opt=opt),
        "AWarp(depth=[8, 4, 4], chroma=1)": lambda opt: core.warp.AWarp(src, mask(src, opt), depth=[8, 4, 4], chroma=1, opt=opt),
        "AWarp() 4x clip": lambda opt: core.warp.AWarp(src4, mask(src, opt), opt=opt),
        "AWarp() 4x clip YUV420P16": lambda opt: core.warp.AWarp(src4_16, mask(src16, opt), opt=opt),
        "AWarpSharp2(chroma=1) 36x28": lambda opt: core.warp.AWarpSharp2(narrow36, chroma=1, opt=opt),
        "AWarpSharp2(thresh=255, chroma=1) 44x28": lambda opt: core.warp.AWarpSharp2(narrow44, thresh=255, chroma=1, opt=opt),
        "ABlur(type=1) 36x28": lambda opt: core.warp.ABlur(narrow36, type=1, opt=opt),
        "ABlur(blur=1, type=0) 36x28 YUV420P16": lambda opt: core.warp.ABlur(narrow36_16, blur=1, type=0, opt=opt),
    }


def main():
    if len(sys.argv) > 1:
        core.std.LoadPlugin(sys.argv[1])
    check(hasattr(core, "warp"), "plugin loaded (namespace warp)")
    for name in ("AWarpSharp2", "ASobel", "ABlur", "AWarp"):
        check(hasattr(core.warp, name), "function %s registered" % name)

    src = pattern_clip(100, 60)

    out = core.warp.AWarpSharp2(src)
    check(out.format.id == src.format.id and out.width == 100 and out.height == 60 and out.num_frames == 1,
          "format, size and length kept")
    changed = sum(a != b for a, b in zip(plane(out, 0), plane(src, 0)))
    check(changed > 100, "luma warped (%d of %d pixels changed)" % (changed, 100 * 60))

    out = core.warp.AWarpSharp2(src, planes=[0])
    check(plane(out, 1) == plane(src, 1) and plane(out, 2) == plane(src, 2), "planes=[0]: chroma copied unchanged")
    out = core.warp.AWarpSharp2(src, planes=[1, 2])
    check(plane(out, 0) == plane(src, 0), "planes=[1, 2]: luma copied unchanged")

    flat = core.std.BlankClip(width=100, height=60, format=vs.YUV420P8, color=[90, 140, 100], length=1)
    sob = core.warp.ASobel(flat)
    check(set(plane(sob, 0)) == {0} and set(plane(sob, 1)) == {0}, "ASobel of a flat clip is 0")
    check(max(plane(core.warp.ASobel(src), 0)) <= 128, "ASobel output limited to thresh (default 128)")
    check(max(plane(core.warp.ASobel(src, thresh=40), 0)) <= 40, "ASobel output limited to thresh=40")

    check(digest(core.warp.ABlur(src, blur=0)) == digest(src), "ABlur(blur=0) returns the input unchanged")
    check(digest(core.warp.AWarpSharp2(flat)) == digest(flat), "AWarpSharp2 of a flat clip is unchanged")

    edge = core.warp.ABlur(core.warp.ASobel(src))
    # depth=0 keeps every pixel except the last row of each plane, which the original algorithm
    # always interpolates with the row above (its vertical limit for the last row is -1).
    aw0 = core.warp.AWarp(src, edge, depth=[0, 0, 0])
    same = all(plane(aw0, p)[:-aw0.width >> (p and 1)] == plane(src, p)[:-src.width >> (p and 1)] for p in range(3))
    check(same, "AWarp(depth=0) keeps the clip unchanged (except the last row)")
    src4 = core.resize.Point(src, 400, 240)
    aw4 = core.warp.AWarp(src4, edge)
    check(aw4.width == 100 and aw4.height == 60, "AWarp with a 4x clip has the mask's size")

    # The output has the mask's length; a shorter clip repeats its last frame.
    mask3 = core.std.Loop(edge, 3)
    aw = core.warp.AWarp(src, mask3)
    check(aw.num_frames == 3 and digest(aw, 2) == digest(aw, 0), "AWarp with a mask longer than the clip")

    for name, make in cases().items():
        d_opt = digest(make(True))
        d_c = digest(make(False))
        check(d_opt == d_c, "opt=True and opt=False agree: %s" % name)
        expected = REFERENCE.get(name)
        if expected is None:
            print("info reference %s: %s" % (name, d_opt))
        else:
            check(d_opt == expected, "reference result: %s" % name)

    expect_error(lambda: core.warp.AWarpSharp2(src, thresh=256), "thresh 256 rejected", "thresh")
    expect_error(lambda: core.warp.AWarpSharp2(src, blur=-1), "blur -1 rejected", "blur")
    expect_error(lambda: core.warp.AWarpSharp2(src, type=2), "type 2 rejected", "type")
    expect_error(lambda: core.warp.AWarpSharp2(src, depth=[128]), "depth 128 rejected", "depth")
    expect_error(lambda: core.warp.AWarpSharp2(src, depth=[16, -129]), "depth -129 rejected", "depth")
    expect_error(lambda: core.warp.AWarpSharp2(src, chroma=2), "chroma 2 rejected", "chroma")
    expect_error(lambda: core.warp.AWarpSharp2(src, cplace="mpeg4"), "cplace mpeg4 rejected", "cplace")
    expect_error(lambda: core.warp.AWarpSharp2(src, planes=[3]), "plane 3 rejected", "plane index")
    expect_error(lambda: core.warp.AWarpSharp2(src, planes=[1, 1]), "plane given twice rejected", "twice")
    expect_error(lambda: core.warp.AWarpSharp2(src.resize.Point(format=vs.RGB24, matrix_in_s="709")), "RGB rejected", "RGB")
    expect_error(lambda: core.warp.AWarpSharp2(src.resize.Point(format=vs.YUV420PS)), "float rejected", "integer")
    expect_error(lambda: core.warp.AWarpSharp2(src.resize.Point(format=vs.YUV411P8)), "4:1:1 rejected", "subsampling")
    expect_error(lambda: core.warp.ASobel(src, thresh=-1), "ASobel thresh -1 rejected", "thresh")
    expect_error(lambda: core.warp.ABlur(src, type=-1), "ABlur type -1 rejected", "type")
    expect_error(lambda: core.warp.AWarp(src, edge.resize.Point(format=vs.YUV420P16)), "AWarp format mismatch rejected", "same format")
    expect_error(lambda: core.warp.AWarp(core.resize.Point(src, 200, 120), edge), "AWarp 2x clip rejected", "four times")
    expect_error(lambda: core.warp.AWarp(src, edge, depth=[127, 0, -129]), "AWarp depth -129 rejected", "depth")

    if FAILURES:
        raise SystemExit("%d check(s) failed" % len(FAILURES))
    print("all tests passed")


if __name__ == "__main__":
    main()
