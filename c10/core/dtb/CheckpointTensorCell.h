#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/core/dtb/Rematerializer.h>

namespace at{
  class Tensor;
}

// TODO: change TORCH_CHECK

namespace c10{
namespace dtb{

using at::Tensor;

struct CheckpointTensorCell : intrusive_ptr_target {
#ifdef DEBUG_MODE
  long id = gen_counter();
  static long counter;
  static long gen_counter() {
    return counter++;
  }
  std::string counter_name(){
    return std::string("x") + std::to_string(id);
  }
#endif
  std::unique_ptr<Tensor> t;
  bool defined = false;         // 标记cell是否存在
  bool is_undefined_tensor;     // 标记是否是空张量
  int degree = 0;
  void add_degree(int deg) { degree += deg; }
  int get_degree() { return degree; }
  DispatchKeySet key_set_;
  DispatchKeySet key_set() const {
    // TORCH_CHECK(defined);
    return key_set_;
  }
  caffe2::TypeMeta dtype_;
  caffe2::TypeMeta dtype() const {
    // TORCH_CHECK(defined);
    return dtype_;
  }
  c10::optional<Device> optional_device_;
  c10::optional<Device> optional_device() const {
    // TORCH_CHECK(defined);
    return optional_device_;
  }
  // A Tensor is evictable iff it's AliasPool is evictable.
  // A evictable tensor must have Rematerializer.
  intrusive_ptr<AliasPool> pool;
  intrusive_ptr<Rematerializer> remat;
  void evict();

  void fill(Tensor& t);

  explicit CheckpointTensorCell(Tensor& t, const intrusive_ptr<AliasPool>& pool);

  explicit CheckpointTensorCell(Tensor& t,
                                const intrusive_ptr<AliasPool>& pool,
                                const intrusive_ptr<Rematerializer>& remat);

  size_t memory() {
    // TORCH_CHECK(defined);
    return pool->memory;
  }
  Tensor get();
  Tensor get(int&);   // for remat count (deprecated)

  void remat_neghibors(int remat_depth);
  int precheck();
  // std::vector<int64_t> sizes(){
  //   return get().sizes().vec();
  // }
  void pin();
  
  void release_resources() override;

};

}    
}

