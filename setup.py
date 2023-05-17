import glob
import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
  this_dir = os.path.dirname(os.path.abspath(__file__))
  extensions_dir = os.path.join(this_dir, "ops")

  main_source = os.path.join(extensions_dir, "vision.cpp")
  sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
  sources_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"))

  sources = [main_source] + sources

  extension = CppExtension

  extra_compile_args = {"cxx": []}
  define_macros = []

  if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
    extension = CUDAExtension
    sources += sources_cuda
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
      "-DCUDA_HAS_FP16=1",
      "-D__CUDA_NO_HALF_OPERATORS__",
      "-D__CUDA_NO_HALF_CONVERSIONS__",
      "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    CC = os.environ.get("CC", None)
    if CC is not None:
      extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

  sources = [os.path.join(extensions_dir, s) for s in sources]

  include_dirs = [extensions_dir]

  ext_modules = [
    extension(
      "ops._C",
      sources,
      include_dirs=include_dirs,
      define_macros=define_macros,
      extra_compile_args=extra_compile_args,
    )
  ]

  return ext_modules


setup(
  name="hsi",
  version="1.0",
  author="Shijie Wang",
  ext_modules=get_extensions(),
  cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
