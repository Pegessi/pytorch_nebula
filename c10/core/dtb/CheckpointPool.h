#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/core/dtb/External.h>
#include <c10/core/dtb/ResidualChain.h>


namespace c10{
namespace dtb{

// CheckpointPool keep a list of AliasPool, and search over them to choose the best one to evict.
struct CheckpointPool : intrusive_ptr_target {
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
  void clear_exts();

};


}
}