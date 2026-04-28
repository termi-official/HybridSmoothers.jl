module HybridSmoothersCUDAExt

using HybridSmoothers
using KernelAbstractions

import CUDA: CUDA, CUDABackend

import HybridSmoothers: CudaDevice
import HybridSmoothers: sparsemat_format_type, CSCFormat, CSRFormat

function HybridSmoothers.default_backend(::CudaDevice)
    return CUDABackend()
end

end
