#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/Rematerializer.h>

namespace c10 {
namespace dtb {
    
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
  bool if_temp = false;           // mark if this pool belongs to a temp_cptc
  bool is_evicted = false;      // if pool is evicted(not in mem)
  bool is_retain = false;       // if proactive lock
  
  int dependency = 0;
  std::future<int> dep_future;
  // lock() && unlock() used for protect storage during tensor operations
  inline void lock() {
    ++lock_count;
  }
  void unlock();

//   void AliasPool::unlock() {
//     --lock_count;   // external == 0 , lock_count > 0 == 0
//     /// improvement for life cycle
//     /// because that staleness is harmful to eviction of remated tensor during backward progress, which should be released immediately
// #ifndef ORIGINAL_DTR
//     if(remat_count>0){
//       unlock_remated();
//     #ifdef DEBUG_MODE
//       if(record_lifecycle){ // 这里记录的是重物化过程的情况
//         pid_t pid = getpid();
//         DTRLogLifeCycle(std::to_string(pid), external_count, lock_count, remat_count);
//       }
//     #endif
//       if(remat_count == 0 && external_count == 0 && lock_count == 0 && retain_count == 0){
//         if (memory > 0 && (!ecn) && head_remat) {
//           evict(1);
//         } 
//         // else if (memory > 0 && head_remat==nullptr)
//         //   evict(2);
//       }
//     }
//     /**
//      * 上面的重物化检查相当于提供了一个释放重物化张量的timing
//      * 但实际上由于动态执行中会出现remat_count==0但lock_count>0导致无法回收的情况（错过了这个回收窗口）
//      * 因此在反向过程中额外检查是否有释放的机会
//     */
//     if(during_backward){
//       if(remat_count == 0 && external_count == 0 && lock_count == 0){
//         if (memory > 0 && (!ecn) && head_remat) {
//           evict(1);
//         } 
//       }
//     }
// #endif
//   }

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

  void clone_and_reset();
  
  intrusive_ptr<Rematerializer> head_remat;
  bool evictable() const {
#ifndef ORIGINAL_DTR
    // return lock_count == 0 && head_remat && remat_count == 0;   // 存在一些没有head_remat的权重转换，如rope的freqs
    return lock_count == 0 && head_remat && remat_count == 0 && !is_retain && !if_weight;
#else
    return lock_count == 0 && head_remat;
#endif
  }

  size_t memory;
  time_t last_used_time;
  uintptr_t addr;               // address of tensor data ptr
  // An aliaspool cant register itself to the checkpointpool - you have to do it yourself.
  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, int device_id);

  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id);

  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id, bool if_w);

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
  void set_addr(uintptr_t new_addr) { 
    // printf("[SET ADDR] %ld to %ld\n", addr, new_addr);
    addr = new_addr; 
  }
  // register_external() && release_external() is used for maintain the aps natural period
  void register_external() {
    ++external_count;
  }
  void release_external();    /// original release trigger
  // if it was evicted, refresh it. otherwise do nothing.
  // have to check so, because when we rematerialize a single tensor in an aliaspool,
  // we will set it to non-evicted, and when we rematerialize it's tensor they will also reset this.
  void set_not_evicted();
  void release_resources() final;
};

}
}