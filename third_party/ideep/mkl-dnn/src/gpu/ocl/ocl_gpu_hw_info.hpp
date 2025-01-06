/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_OCL_GPU_HW_INFO_HPP
#define GPU_OCL_OCL_GPU_HW_INFO_HPP

#include <CL/cl.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/device_info.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

void init_gpu_hw_info(engine_t *engine, cl_device_id device, cl_context context,
        compute::gpu_arch_t &gpu_arch, int &stepping_id,
        bool &mayiuse_ngen_kernels);

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_GPU_HW_INFO_HPP
