from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lcc_cuda',
    ext_modules=[
        CUDAExtension('lcc_cuda', [
            'lcc_cuda.cpp',
            'lcc_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

