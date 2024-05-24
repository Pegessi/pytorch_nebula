#pragma once


#include <iostream>
#include <vector>
#include <atomic>
#include <memory>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/flat_hash_map.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/core/dtb/CheckpointPool.h>
#include <c10/core/dtb/External.h>
#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/core/dtb/ResidualChain.h>
#include <c10/core/dtb/MemGraph.h>

namespace c10 {
namespace dtb {

static const bool USE_DTR = ([]() -> bool {    /// init if use dtr by check env DTR_ENABLE
    const char* env = getenv("DTR_ENABLE");
    if(env) return atoi(env)==1;
    else return false;
})();

  size_t current_memory(int device = 0);

  size_t reserved_memory(int device = 0);

  class DTBCheckpointPool{
    private:
      std::vector<std::unique_ptr<CheckpointPool>> device_dtbpool;
      std::vector<size_t> peak_allocated_memory;
      std::vector<size_t> peak_reserved_memory;
#ifdef MEM_FIRST_EVICT
      ska::flat_hash_map<void*, weak_intrusive_ptr<AliasPool>> p2ap;
      // std::unordered_map<void*, weak_intrusive_ptr<AliasPool>> p2ap;
#endif

      void update_max_meminfo(int device_id){
        peak_allocated_memory[device_id] = std::max(peak_allocated_memory[device_id], current_memory(device_id));
        peak_reserved_memory[device_id] = std::max(peak_reserved_memory[device_id], reserved_memory(device_id));
      }

      bool device_id_check(int device_id){
        return device_id >= 0;
      }

      inline void init_check(){
        if(!initialized())
        {
        #ifdef DEBUG_MODE
          printStackTrace();
        #endif
          throw std::runtime_error("DTB manager is not initialized.");
        }
      }

#ifdef MEM_FIRST_EVICT
      void insert_ptr2ap(void* ptr, weak_intrusive_ptr<AliasPool>&& ap) {
        // printf("[INSERT CHECK] %ld\n", reinterpret_cast<uintptr_t>(ptr));
        if(likely(ptr))   // some times with undefined tensor which memory=0
        {
          auto it = p2ap.find(ptr);
          // TORCH_INTERNAL_ASSERT(it==p2ap.end());
          if(likely(it!=p2ap.end()))  // 是否是同一个pool？
            erase_ptr2ap(ptr);
          p2ap.insert({ptr, ap});
          
        }else{
          // printf("[ABNORMAL AP] ptr:%ld defined:%d", reinterpret_cast<uintptr_t>(ptr), ap->is_evicted ? 0 : 1);
          // printf("[ABNORMAL AP] ptr:%ld defined:%d", reinterpret_cast<uintptr_t>(ptr), ap.defined());
        }
          // p2ap.insert({ptr, ap});
      }

      void erase_ptr2ap(void* ptr) {
        TORCH_INTERNAL_ASSERT(ptr);
        auto it = p2ap.find(ptr);
        if(likely(it!=p2ap.end()))
          p2ap.erase(it);
      }
#endif      

    public:
      std::vector<bool> if_train_mode;
      std::vector<bool> if_during_backward;

      void init(int device_count);

      bool initialized() {
        return !device_dtbpool.empty();
      }

      void auto_evict(int device);

      bool auto_evict(int device, size_t coming_bytes);

      void force_evict(int device, int mode);

      void add_into_keychain(int device, const weak& new_key, const weak& pre);

      void add_ap(int device, intrusive_ptr<AliasPool>& new_ap);

      void add_ext(int device, const weak_intrusive_ptr<External>& new_ext);

      void erase_ap(int device, uintptr_t addr);

#ifdef MEM_FIRST_EVICT
      void update_ap(intrusive_ptr<AliasPool>& ap, uintptr_t new_addr);

      void remove_p2ap(uintptr_t addr);

      std::pair<weak_intrusive_ptr<AliasPool>, bool> get_ap_by_ptr(void* ptr);

      size_t get_p2ap_size() {return p2ap.size();}

      bool check_ptr_in_aps(int device, uintptr_t addr);

      size_t get_aps_size(int device);
#endif

      void toggle_sampling(bool if_sampling);

      void toggle_ignore_small_tensors(bool if_ignore);

      void set_memory_budget(long budget);

      void unset_memory_budget();

      void set_train_mode(bool flag);

      void set_during_backward(bool flag);

      void clear_checkpointpool(int device);

      void pool_cur_mem_snapshot(int device);

      std::vector<std::pair<size_t, size_t>> get_peak_memory();
  };

  // extern DTBCheckpointPool dtb_pool;
  C10_CUDA_API extern std::atomic<DTBCheckpointPool*> PoolManager;

  inline DTBCheckpointPool* get() {
    return PoolManager.load();
  }

  void init(int device_count);

  void dtbPoolInitEntry();

  void lazyInitDTB();

  /// use pool with call: auto* poolManager = getDTBPoolManager();
  c10::dtb::DTBCheckpointPool* getDTBPoolManager();

}
}