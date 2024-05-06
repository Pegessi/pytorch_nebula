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
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define TORCH_CHECK(a, ...) // profile mode

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

#pragma region UnionSet

// TODO: using a pool allocator might make more sense - no need to allocate and delete each pointer individually.
template<typename T>
struct EquivalentClassNode : intrusive_ptr_target {
  explicit EquivalentClassNode(const T& t) : t_unsafe(t) { }
  mutable intrusive_ptr<EquivalentClassNode> parent;
  bool is_root() {    // no parent node means root
    return !parent;
  }
  void release_resources() override {
    parent.reset();
  }
  T t_unsafe;
};

// 返回根节点的值
template<typename T>
T& get_t(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  return find_root(n)->t_unsafe;
}

template<typename T>
static void update_t(const intrusive_ptr<EquivalentClassNode<T>>& n, const T& t) {
  find_root(n)->t_unsafe = t;
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> find_root(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  if (n->is_root()) {
    return n;
  } else {
    n->parent = find_root(n->parent);
    return n->parent;
  }
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> merge(const std::function<T(const T&, const T&)>& merge_t,
                                            const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                                            const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto l = find_root(lhs);
  auto r = find_root(rhs);
  if (l == r) {
    return l;
  }
  l->parent = r;
  r->t_unsafe = merge_t(l->t_unsafe, r->t_unsafe);
  return r;
}

template<typename T>
bool is_same_union(const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                   const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto l = find_root(lhs);
  auto r = find_root(rhs);
  return l == r;
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> flat(const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                                            const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto root = find_root(lhs);
  lhs->parent = root;
  rhs->parent = root;
  return root;
}

#pragma endregion

size_t memory(const Tensor& t);
size_t get_addr(const Tensor& t);

template<typename T>
struct RefCell final : intrusive_ptr_target {
  mutable T value;
  void release_resources() final {
    static_release_resources(value);
  }
  RefCell(const T& t) : value(t) { }  /// TODO: 这里没有std::move，传入的T是c10::intrusive_ptr<at::External, c10::detail::intrusive_target_default_null_type<at::External> >
};

template<typename T>
using Ref = intrusive_ptr<RefCell<T>>;

template<typename T>
void static_release_resources(intrusive_ptr<T>& ptr) {
  ptr.reset();
}

class CheckpointTensorCell;
using strong = intrusive_ptr<CheckpointTensorCell>;
using strongs = std::vector<strong>;
using weak = weak_intrusive_ptr<CheckpointTensorCell>;
using weaks = std::vector<weak>; 
using Tensors = std::vector<Tensor>;
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;
using mutate_function_t = std::function<void(const Tensors&)>;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;
struct CheckpointInfo {
  duration_t compute_cost;
  // Floating Point instability?
  double cost(size_t memory, size_t staleness) const {
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  double cost(size_t memory, size_t staleness, uintptr_t addr) const {  /// TODO: 设计一个新的计算指标
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  CheckpointInfo(duration_t compute_cost) :
    compute_cost(compute_cost) {
  }
};

// ecn represent a evicted tensor group.
// it is a set of tensor that are evicted, and if two evicted tensor are input -> output to each other,
// they must be in an ecn.
// note: we try to support removal from ecn by subtracting compute_cost and memory.
// this will create suprious connection but that should be fine empircally.
// below is an example of a suprious connection:
// a -> b, a -> c
// a, b, c got evicted so belong to a single ecn.
// a got rematerialized.
// b, c still belong to a single ecn although there is no connection.
using ecn_ptr = intrusive_ptr<EquivalentClassNode<CheckpointInfo>>;

struct Unsafe { };

// The rematerializer could be called to reinvoke an operator.
// Tensor point to remat which point to Tensor.
// To build the cycle remat support a default constructor,
// And allow you to fill in the member later.
struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t func;
  strongs inputs;
  weaks outputs;
  duration_t compute_cost;
  int64_t rid;   // remat func fingerprint
  // when some output in here get evicted, they should belong to this ecn.
  // a rematerializer have to track this,
  // because when multiple output of a rematerializer get evicted,
  // we only want to count the compute cost once.
  ecn_ptr ecn;
  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs,
                 duration_t compute_cost)  :
    func(func),
    inputs(inputs),
    compute_cost(compute_cost) {
  }
  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs,
                 int64_t rid,
                 duration_t compute_cost)  :
    func(func),
    inputs(inputs),
    rid(rid),
    compute_cost(compute_cost) {
  }
  void release_resources() final {
    func = rematerialize_function_t();
    inputs.clear();
    outputs.clear();
  }
  void remat();
  void remat(int&);
  ecn_ptr get_ecn();
  CheckpointInfo get_cpi();
};

// Track all Tensor that share the same Storage.
// This is the atomic level of eviction - when evicting, everything here will get evicted.
// When an AliasPool is evicted, the Storage of the underlying tensor must be freed.
// Additionally, the AliasPool contain weak pointer to all children of tensors,
// in order to compute the score of evicting a Storage.
struct AliasPool : intrusive_ptr_target {
  int device_id;
  weaks tensors;
  weaks neighbors;
  std::set<ecn_ptr> neighbor_ecn();
  // get() might hold some raw Tensor, rendering them unevictable.
  // it is likely that get() will run out of memory, and when it does so, it will try to evict.
  // so, it is crucial that we dont try to evict those tensors - doing so will not evict anything.
  // lock_count count how many time a tensor is referenced by get.
  /// triple ref counts for management of aps life cycle
  size_t lock_count = 0;          // for get() call, which is used in remat() and make_raw()
  size_t external_count = 0;      // for original life cycle of pytorch tensor
  size_t remat_count = 0;         // for remat() call, which is used for improving the life cycle during backward
  size_t retain_count = 0;        // for retain long remat, |disabled| performance worse
  bool if_weight = false;         // flag for mark the tensors transformerd from weights

  int dependency = 0;
  std::future<int> dep_future;
  // lock() && unlock() used for protect storage during tensor operations
  inline void lock() {
    ++lock_count;
  }
  inline void unlock();
  inline void lock_remated(){
    ++remat_count;
    // remat_count = 1;
  }
  inline void unlock_remated(){
    --remat_count;
    // remat_count = 0;
  }
  inline void lock_retain(){
    retain_count++;
  }
  
  intrusive_ptr<Rematerializer> head_remat;
  bool evictable() const {
#ifndef ORIGINAL_DTR
    return lock_count == 0 && head_remat && remat_count == 0;   // 存在一些没有head_remat的权重转换，如rope的freqs
#else
    return lock_count == 0 && head_remat;
#endif
  }
  // if it is not evictable it must not be evicted.
  bool is_evicted = false;
  bool is_retain = false;
  size_t memory;
  time_t last_used_time;
  uintptr_t addr;               // address of tensor data ptr
  // An aliaspool cant register itself to the checkpointpool - you have to do it yourself.
  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, int device_id) :
    head_remat(head_remat),
    memory(memory),
    device_id(device_id),
    last_used_time(std::chrono::system_clock::now()) {
  }
  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id) :
    head_remat(head_remat),
    memory(memory),
    addr(addr),
    device_id(device_id),
    last_used_time(std::chrono::system_clock::now()) {
  }
  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id, bool if_w) :
    head_remat(head_remat),
    memory(memory),
    addr(addr),
    device_id(device_id),
    if_weight(if_w),
    last_used_time(std::chrono::system_clock::now()) {
  }
  // if it is evicted, then hold the evicted tensor group.
  ecn_ptr ecn;
  double cost(time_t current_time);
  void evict(int mode=0);
  int update_dep_task();
  void update_dependency();
  int get_dependency() { 
    if(dep_future.valid()){
      dependency = dep_future.get(); 
    }
    return dependency;
  }
  void set_addr(uintptr_t new_addr) { addr = new_addr; }
  // register_external() && release_external() is used for maintain the aps natural period
  void register_external() {
    ++external_count;
  }
  void release_external();    /// original release trigger
  // if it was evicted, refresh it. otherwise do nothing.
  // have to check so, because when we rematerialize a single tensor in an aliaspool,
  // we will set it to non-evicted, and when we rematerialize it's tensor they will also reset this.
  void set_not_evicted(const intrusive_ptr<AliasPool>& self);
  void release_resources() final {
    tensors.clear();
    neighbors.clear();
    head_remat.reset();
  }
};

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
    TORCH_CHECK(defined);
    return key_set_;
  }
  caffe2::TypeMeta dtype_;
  caffe2::TypeMeta dtype() const {
    TORCH_CHECK(defined);
    return dtype_;
  }
  c10::optional<Device> optional_device_;
  c10::optional<Device> optional_device() const {
    TORCH_CHECK(defined);
    return optional_device_;
  }
  // A Tensor is evictable iff it's AliasPool is evictable.
  // A evictable tensor must have Rematerializer.
  intrusive_ptr<AliasPool> pool;
  intrusive_ptr<Rematerializer> remat;
  void evict() {
    TORCH_CHECK(remat);
    defined = false;
    t.reset();
  }
  void fill(Tensor& t);

  explicit CheckpointTensorCell(Tensor& t, const intrusive_ptr<AliasPool>& pool) : pool(pool) {
    fill(t);
  }
  explicit CheckpointTensorCell(Tensor& t,
                                const intrusive_ptr<AliasPool>& pool,
                                const intrusive_ptr<Rematerializer>& remat) :
    pool(pool), remat(remat) {
    fill(t);
  }

  size_t memory() {
    TORCH_CHECK(defined);
    return pool->memory;
  }
  Tensor get();
  Tensor get(int&);   // for remat count (deprecated)
  int precheck();
  // std::vector<int64_t> sizes(){
  //   return get().sizes().vec();
  // }
  void pin() {
    // get();         // [TAG] this is for debug to find out tensors unreleased
    pool->head_remat.reset();
    remat.reset();
  }
  void release_resources() final {
    defined = false;
    t.reset();
    pool.reset();
    remat.reset();
  }
};

// An external reference.
// Each strong will have at most one external reference.
// By keeping such an invariant, whenever an external reference die,
// We know that the underlying strong is only used internally.
// Thus, when it die we can apply optimization like banishing/infinite staleness.
// We keep this invariant by only allowing CheckpointTensorImpl to make new External,
// When new CheckpointTensorImpl is constructed.
struct External : intrusive_ptr_target {
  External(const strong& value) : value(value) {
    value->pool->register_external();                       /// TAG: Aliaspool引用计数的唯一增加入口
  }
  External(Tensor& value, bool if_weight=false) :
    External(strong::make(  // const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id, bool if_w
                          value,
                          intrusive_ptr<AliasPool>::make(  /// [TAG] AliasPool构造
                            Unsafe(),                        
                            intrusive_ptr<Rematerializer>(),
                            memory(value),
                            get_addr(value),
                            value.defined() ? static_cast<int>(value.device().index()) : -1,
                            if_weight)
                          )
            ) {}
          /// static_cast<int>(value.device().index()) 存在无device的tensor, probably empty tensor
  External(Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat) :
    External(strong::make(value, pool, remat)) { }
  strong value;
  void release_resources() override{    /// TAG: Aliaspool引用计数的唯一减少入口
      // printf("%s %d %ld %d ex:%ld\n", value->counter_name().c_str(), ((value->pool->memory > 0 && (!value->pool->ecn) && value->pool->head_remat)||(value->pool->memory > 0&& value->pool->head_remat==nullptr && !value->pool->if_weight)) ? 1 : 0, value->pool->memory, value->pool->if_weight ? 1 : 0, value->pool->external_count);
      value->pool->release_external();
      // printf("pool of %s release_external finish.\n", value->counter_name().c_str());
      value.reset();
  }
};

#ifdef DEBUG_MODE
void printStackTrace();
#endif

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

  void* mutable_data_cpti() {
    return unsafeGetTensorCell()->t->mutable_data_ptr();
  }

  const void* data_cpti() {
    return unsafeGetTensorCell()->t->data_ptr();
  }

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


enum KeyChainStatus {
  IS_RIGHT_CHAIN,   // 是要找的链
  TO_BE_DELETE,     // 需要删除该链
  NORMAL            // 正常状态
};

struct ChainNode;
using StrongChainNode = intrusive_ptr<ChainNode>;
using WeakChainNode = weak_intrusive_ptr<ChainNode>;

struct ChainNode : intrusive_ptr_target {
  weak value;
  bool is_lock = false;
  mutable intrusive_ptr<ChainNode> parent;
  ChainNode(const weak& weak_cell) : value(weak_cell) {}
  bool is_equal(const StrongChainNode& other) const {
      return value == other->value;
  }
  void lock_value(){
    if(!is_lock){
      if(auto cell = value.lock()){
        cell->get();
        cell->pool->is_retain = true;
        cell->pool->lock();
        is_lock = true;
      }
    }
  }
};

constexpr const int CHAIN_LENGTH_LOCK_THRESHOLD = 16;
constexpr const int CHAIN_LOCK_STRIDE = 2;

// TODO: weak并不能作为键
struct ResidualChain : intrusive_ptr_target {
  StrongChainNode root;
  std::vector<StrongChainNode> members;
  int last_check_idx = 0;

  ResidualChain(const StrongChainNode& n) : root(n) {
    members.push_back(n);
  }

  size_t size(){
    return members.size();
  }

  void insert(const StrongChainNode& n) {
    n->parent = root;
    members.push_back(n);

    if(size()>CHAIN_LENGTH_LOCK_THRESHOLD) {  // 认为该链是要找的链
      for(int i = last_check_idx; i<size(); i++){
        if(i%CHAIN_LOCK_STRIDE==0)
          members[i]->lock_value();
      }
      last_check_idx = size() - 1;
    }
  }

  void erase(const StrongChainNode& n) {
    auto it = std::find(members.begin(), members.end(), n);
    if(it!=members.end())
      members.erase(it);
  }

  bool in_chain(const StrongChainNode& n){
    const auto& last_node = members.back();
    return last_node->is_equal(n);
  }

  void release_resources() override {
    members.clear();
    root.reset();
  }
};

using ResidualChainRef = intrusive_ptr<ResidualChain>;

// CheckpointPool keep a list of AliasPool, and search over them to choose the best one to evict.
struct CheckpointPool {
  std::vector<weak_intrusive_ptr<AliasPool>> aps;
  std::map<uintptr_t, weak_intrusive_ptr<AliasPool>> mem_ordered_aps;

  std::vector<weak_intrusive_ptr<External>> exts;
  std::vector<weak> candidates;           // candidates for end point
  std::vector<ResidualChainRef> chains;

  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
  // whether to take a square-root sample of the pool during an eviction loop
  bool sample_tensors = true;
  // ignore tensors < 1% of the average tensor size
  bool ignore_small_tensors = true;
  bool has_memory_budget = false;
  long memory_budget;
  void evict();
  void evict(size_t need_to_free_bytes);
  void mem_first_evict(bool&);
  void exec_first_evict();
  
  void auto_evict();
  void auto_evict(size_t size_bytes);
  /// for early check and evict
  /// for initiative evict
  void force_evict(int mode);
  void initiative_evict(size_t to_free_bytes);
  
  void add(const intrusive_ptr<AliasPool>&);
  // void add_chain(const intrusive_ptr<KeyChain>&);
  // void erase_chain(intrusive_ptr<KeyChain>&);
  CheckpointPool();

  /// functional
  void clear_checkpointpool();
  void set_sample_tensors(bool flag){
    sample_tensors = flag;
  }
  void set_ignore_small_tensors(bool flag){
    ignore_small_tensors = flag;
  }
  void set_memory_budget(long budget){
    memory_budget = budget;
    has_memory_budget = true;
  }
  void unset_memory_budget(){
    has_memory_budget = false;
  }
  void clear_exts(){
    candidates.clear();
    chains.clear();
    int count = 0;
    while (!exts.empty()) {
      if (auto e = exts.back().lock()) {
        e->value->pin();  /// why pin and remat?
        // if((e->value->pool->lock_count!=0||e->value->pool->external_count>0||e->value->pool->remat_count>0)&&e->value->defined){
        //   count++;
        //   // printf("release meet locked %d\n", count);
        //   printf("remat_count:%ld, external_count:%ld, lock_count:%ld, counts:%d\n", e->value->pool->remat_count, e->value->pool->external_count, e->value->pool->lock_count, count);
        // }
      }
      exts.pop_back();
    }
  }
};

// inline CheckpointTensorImpl* get_cpti(const Tensor& t) {
//   auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
//   if(cpti==nullptr){  // 存在输入是在op内部创建的tensor，其并不是cpti
    
//   }else{
//     TORCH_CHECK(cpti != nullptr);
//     return cpti;
//   }
// }

// inline Ref<intrusive_ptr<External>> cell_from_tensor(const Tensor& t) {
//   return get_cpti(t)->ref;
// }

}
