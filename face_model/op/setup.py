from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
module_path = os.path.dirname(__file__)
setup(name='customops', version='0.0.1', 
      ext_modules=[cpp_extension.CUDAExtension(name="fused",
                            sources=["fused_bias_act.cpp", "fused_bias_act_kernel.cu"], include_dirs=cpp_extension.include_paths(),),
                   cpp_extension.CUDAExtension(name="upfirdn2d_op",
                             sources=["upfirdn2d.cpp", "upfirdn2d_kernel.cu"],include_dirs=cpp_extension.include_paths(),),
                   ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})