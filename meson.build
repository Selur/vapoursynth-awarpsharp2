project('AWarpSharp2', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.64.0',
  version : '4.2.0'
)

sources = ['src/awarpsharp2.cpp']

# Prefer system VapourSynth headers, fall back to the vendored API 4 headers.
system_vs_dep = dependency('vapoursynth', version : '>=55', required : false)
if system_vs_dep.found()
  vapoursynth_dep = system_vs_dep.partial_dependency(compile_args : true, includes : true)
else
  vapoursynth_dep = declare_dependency(include_directories : include_directories('include/vapoursynth'))
endif

cxx = meson.get_compiler('cpp')
msvc = cxx.get_argument_syntax() == 'msvc'
if not msvc
  add_project_arguments('-Wall', '-Wextra', '-Wshadow', language : 'cpp')
endif

# x86 gets the SSE2 code and arm64 the NEON code next to the C code (opt=True selects it at
# runtime); every other CPU builds the C code only.
if host_machine.cpu_family() in ['x86', 'x86_64']
  add_project_arguments('-DAWARPSHARP2_X86=1', language : 'cpp')
  sources += 'src/sse2.cpp'
  if not msvc
    add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
  endif
elif host_machine.cpu_family() == 'aarch64'
  # NEON is part of the AArch64 base ISA, no extra compiler flag needed.
  add_project_arguments('-DAWARPSHARP2_ARM=1', language : 'cpp')
  sources += 'src/neon.cpp'
endif

if host_machine.system() in ['windows', 'cygwin'] and host_machine.cpu_family() == 'x86' and not msvc
  add_project_arguments('-mstackrealign', language : 'cpp')
  add_project_link_arguments('-Wl,--kill-at', language : 'cpp')
endif

# VapourSynth only autoloads *.dylib on macOS, while Meson defaults to *.so for modules.
module_kwargs = {}
if host_machine.system() == 'darwin'
  module_kwargs += {'name_suffix' : 'dylib'}
endif

if get_option('python_package')
  # Same layout as the VapourSynth PyPI package autoloads from: site-packages/vapoursynth/plugins
  py = import('python').find_installation(pure : false)
  install_dir = join_paths(py.get_install_dir(), 'vapoursynth', 'plugins')
else
  libdir = join_paths(get_option('prefix'), get_option('libdir'))
  if system_vs_dep.found()
    libdir = system_vs_dep.get_variable(pkgconfig : 'libdir', default_value : libdir)
  endif
  install_dir = join_paths(libdir, 'vapoursynth')
endif

shared_module('awarpsharp2', sources,
  dependencies : vapoursynth_dep,
  install : true,
  install_dir : install_dir,
  gnu_symbol_visibility : 'hidden',
  kwargs : module_kwargs
)
