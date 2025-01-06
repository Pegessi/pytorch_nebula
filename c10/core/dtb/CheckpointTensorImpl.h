#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/External.h>
#include <c10/core/TensorImpl.h>

namespace c10{
namespace dtb{

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  #ifdef DEBUG_MODE
  if(t.has(DispatchKey::Checkpoint)) 
  {
    printStackTrace();
    // return t;
  }
  #endif
  CHECK(!t.has(DispatchKey::Checkpoint));
  auto ret = t.add(DispatchKey::Checkpoint);
  return ret;
}

struct TORCH_API CheckpointTensorImpl : public TensorImpl {
  std::string counter_name() const {
#ifdef DEBUG_MODE
    return std::string("x") + std::to_string(ref->value->value->id);
#else
    return std::string("Tensor id records visible only in DEBUG_MODE.");
#endif
  }

  size_t counter_id() const {
#ifdef DEBUG_MODE
    return ref->value->value->id;
#else
    return 0;
#endif
  }

  Ref<intrusive_ptr<External>> ref;

  // void* mutable_data_cpti() {
  //   return unsafeGetTensorCell()->t->mutable_data_ptr();
  // }

  // const void* data_cpti() {
  //   return unsafeGetTensorCell()->t->data_ptr();
  // }

  strong unsafeGetTensorCell(){
    return ref->value->value;
  }

  void release_resources() override;

  // All of constructor will call this
  explicit CheckpointTensorImpl(const Ref<intrusive_ptr<External>>& ref) :
    TensorImpl(convert_key_set(ref->value->value->key_set()),                   // [TAG] 这里添加了checkpoint后端dispatchkey
               ref->value->value->dtype(),
               ref->value->value->optional_device()),
    ref(ref) {
      // ref->value->value == CheckpointTensorCell*
      // mutable_data_func = [this] { return this->mutable_data_cpti(); };      /// [TAG] 注释这里就会让cptc无法被直接访问，这里通过篡改自定义的mutable_data_func实现了子类访问
      // device_opt_ = unsafeGetTensorCell()->t->device();
      set_storage_access_should_throw();
      if(!ref->value->value->defined){
        ref->value->value->get();
      }
      if (key_set().has(DispatchKey::Autograd)) {
        if(ref->value->value->t.get()->requires_grad())
          set_requires_grad(true);
    }
  }

  /**
   * 在make过程中可能会有undefined tensor出现，需要检查
  */
  explicit CheckpointTensorImpl(const intrusive_ptr<External>& e) :
    CheckpointTensorImpl(Ref<intrusive_ptr<External>>::make(e)) {
      if(ref->value->value->get().defined())
        set_sizes_and_strides(ref->value->value->get().sizes(), ref->value->value->get().strides());
    }

  explicit CheckpointTensorImpl(Tensor& t, bool if_weight=false);

  static bool if_weight_;

  static Tensors make(const std::string& name,
                      const rematerialize_function_t& remat,
                      Tensors& inputs);

  static Tensors make(const std::string& name,
                      const rematerialize_function_t& remat,
                      Tensors&& inputs);

  // mutate_idx indicate which of the inputs will get mutated.
  /// TODO: 左值引用和右值引用都是接收的vector，是原输入的副本，针对副本的修改(register)是不能影响到原输入的
  static void mutate(const std::string& name,
                     const mutate_function_t& mutate,
                     Tensors& inputs,
                     const std::vector<size_t>& mutate_idx);
  static void mutate(const std::string& name,
                     const mutate_function_t& mutate,
                     Tensors&& inputs,
                     const std::vector<size_t>& mutate_idx);
  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;
  //////////// this function is private, cannot be changed
  // template <typename VariableVersion>
  // c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
  //     VariableVersion&& version_counter,
  //     bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  int64_t dim() const {
    return ref->value->value->get().dim();
  }
  int64_t numel() const {
    return ref->value->value->get().numel();
  }
  IntArrayRef sizes() const {
    return ref->value->value->get().sizes();
  }
  int64_t size(int64_t d) const {
    return ref->value->value->get().size(d);
  }
  IntArrayRef strides() const {
    return ref->value->value->get().strides();
  }
  int64_t stride(int64_t d) const {
    return ref->value->value->get().stride(d);
  }
  bool has_storage() const override {
    return false;
  }

  ~CheckpointTensorImpl() override;

  //////////////////////////////////// addition ////////////////////////////
  /**
   * 需要说明的是，这里的impl并没有包含storage，意味着不能通过cpti构造的tensor来正常进行操作
   * 而所有需要进入Checkpoint后端的操作是需要实现对应的kernel的，也就是所有使用的op
   * 必须在Checkpoint.cpp中实现对应的warpper，并在native_function.yaml中
  */
  // void refresh_numel() {
  //   TensorImpl::safe_refresh_numel();
  // }
  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  // static void copy_tensor_metadata(
  //     const CheckpointTensorImpl* src_impl,
  //     CheckpointTensorImpl* dest_impl,
  //     const c10::VariableVersion& version_counter,
  //     bool allow_tensor_metadata_change) {
  //   TensorImpl::copy_tensor_metadata(
  //       src_sparse_impl,
  //       dest_sparse_impl,
  //       version_counter,
  //       allow_tensor_metadata_change);

  //     // Sparse-specific fields
  //     dest_impl->sparse_dim_ = src_impl->sparse_dim();
  //     dest_impl->dense_dim_ = src_impl->dense_dim();
  //     dest_impl->indices_ = src_impl->indices();
  //     dest_impl->values_ = src_impl->values();
  //     dest_impl->coalesced_ = src_impl->coalesced();
  //   }

  //   const char* tensorimpl_type_name() const override;
  // };
};


}
}