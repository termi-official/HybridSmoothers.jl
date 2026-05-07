module HybridSmoothersGPUArraysExt

using HybridSmoothers
using GPUArrays

import HybridSmoothers: HybridSmoothers, CSCFormat, CSRFormat

HybridSmoothers.colvals(A::GPUSparseDeviceMatrixCSR) = A.colVal
HybridSmoothers.getrowptr(A::GPUSparseDeviceMatrixCSR) = A.rowPtr

HybridSmoothers.sparsemat_format_type(::GPUSparseDeviceMatrixCSR) = CSRFormat()
HybridSmoothers.sparsemat_format_type(::GPUSparseDeviceMatrixCSC) = CSCFormat()

end
