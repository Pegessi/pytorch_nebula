#pragma once

#include <atomic>
#include <memory>
#include <numeric>
#include <random>
#include <future>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>
#include <c10/core/TensorImpl.h>

#include <c10/core/dtb/AliasPool.h>
#include <c10/core/dtb/Rematerializer.h>
#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/core/dtb/CheckpointTensorImpl.h>
#include <c10/core/dtb/External.h>
#include <c10/core/dtb/ResidualChain.h>
#include <c10/core/dtb/CheckpointPool.h>

// #include <ATen/Tensor.h>
// #include <ATen/ATen.h>

// #define likely(x)      __builtin_expect(!!(x), 1)
// #define unlikely(x)    __builtin_expect(!!(x), 0)
#define TORCH_CHECK(a, ...)   // replace original TORCH_CHECK  profile mode

// #define ORIGINAL_DTR
// #define DEBUG_MODE

// System Description:
// Every Tensor is managed by a CheckpointTensor,
// that describe how it is computed, (the function and the inputs)
// And might optionally hold the tensor value.
// The tensor value might be dropped, and when requested later, recomputed and cached again.

// Corner Cases:
// A CheckpointedTensor might require_grad.
//   In this case the underlying data must not require_grad,
//   as we want backpropagation on the outer, uncheckpoined level.
//   To be more specific, suppose a tensor is recomputed multiple times.
//   We want to only compute the gradient exactly once.
//   To do this, the wrapper must be require_grad, and the wrapped value must not.
// A CheckpointedTensor might be constant.
//   In this case it is unevictable.
// An operator might return multiple output.
//   In this case the computation info (rematerializer) is shared between all of them,
//   And when the function get computed again all value get cached.
// An operator might not return value, but only mutate input value.
//   To combat this, we COW the operator, and wrap CheckpopintTensor with a Ref.
//   By doing this the inner CheckpointTensor is kept purely functional.
// An operator might try to mutate uncheckpointed tensor.
//   We do not support this and will error.
// An operator might create aliases.
//   We track alias in AliasPool.
//   Each AliasPool hold a set of tensor that is alias to eachother.
// An operator might try to create Alias to an unevictable tensor.
//   In such a case the output tensor is unevictable.
// An operator might try to mutate Tensor with Alias.
//   We do not support this case an will error if a Tensor has any alive Alias.
//   However it could be done without a major redesign of the system -
//   Each AliasPool will hold weak pointers to the External Reference.
//   When alias mutation occur,
//   we make a rematerialize_function that take in the base tensor (other tensor alias from)
//   and output all the new value of the aliases, then update the Ref.
//   Of course, the cleaner way is to not support this.
//   Shame on those who use this feature.

// Memory Safety:
// The objects here will have lots of backedges.
// In order to collect memory when computation is completed,
// We require that all strong pointer is of the form of value -> input.
// This ensure that everything will be released if there is no external ref whatsoever.

// Optimization:
// We treat tensor that has no external reference differently -
// They will never be externally used again so we assume their next use time is infinite
// so, if it doesnt has any evicted neighbor it will get evicted immediately.

// Note: to code fast I do not use RAII and just assume the code will not try to recover from exception.
// It should be easy to fix though.

namespace at {

using c10::dtb::strong;
using c10::dtb::strongs;
using c10::dtb::weak;
using c10::dtb::weaks; 
using Tensors = std::vector<Tensor>;
using c10::dtb::rematerialize_function_t;
using c10::dtb::mutate_function_t;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

using c10::dtb::AliasPool;
using c10::dtb::Rematerializer;
using c10::dtb::CheckpointPool;
using c10::dtb::CheckpointTensorCell;
using c10::dtb::CheckpointTensorImpl;
using c10::dtb::External;
using c10::dtb::Unsafe;

}
