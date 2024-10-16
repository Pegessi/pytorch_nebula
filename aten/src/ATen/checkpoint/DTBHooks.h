#pragma once

#include <ATen/detail/CUDADTBHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at { namespace dtb { namespace detail {

// Set the callback to initialize Magma, which is set by
// torch_cuda_cu. This indirection is required so magma_init is called
// in the same library where Magma will be used.
TORCH_CUDA_CPP_API void set_magma_init_fn(void (*magma_init_fn)());


// The real implementation of CUDAHooksInterface
struct DTBHooks : public at::DTBHooksInterface {
  DTBHooks(at::DTBHooksArgs) {}
  void initDTB() const override;
  DeviceIndex current_device() const override;
  int getNumGPUs() const override;
};

}}} // at::dtb::detail
