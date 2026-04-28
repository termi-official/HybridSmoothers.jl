module HybridSmoothersGPUArraysExt

using HybridSmoothers
using GPUArrays

import HybridSmoothers: HybridSmoothers, CSCFormat, CSRFormat

HybridSmoothers.colvals(A::GPUSparseDeviceMatrixCSR) = A.colVal
HybridSmoothers.getrowptr(A::GPUSparseDeviceMatrixCSR) = A.rowPtr

HybridSmoothers.sparsemat_format_type(::GPUSparseDeviceMatrixCSR) = CSCFormat()
HybridSmoothers.sparsemat_format_type(::GPUSparseDeviceMatrixCSC) = CSRFormat()

end
