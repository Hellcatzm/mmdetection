# author: hellcatzm
# data:   2019/7/12

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cc_attention',
    ext_modules=[
        CUDAExtension('cca_cuda', [
            'src/cca_cuda.cpp',
            'src/cca_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})