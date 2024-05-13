#pragma once
#include <c10/cuda/dtb/DTBManager.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>

namespace c10 {
namespace dtb {

size_t current_memory(int device) {
  auto device_stat = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  return device_stat.allocated_bytes[0].current;
}

size_t reserved_memory(int device){
  auto device_stat = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  return device_stat.reserved_bytes[0].current;
}

void DTBCheckpointPool::init(int device_count) {
  const auto size = static_cast<int64_t>(device_dtbpool.size());
  if (size < device_count) {
    device_dtbpool.resize(device_count);
    peak_allocated_memory.resize(device_count);
    peak_reserved_memory.resize(device_count);
    if_train_mode.resize(device_count);
    if_during_backward.resize(device_count);
    for (const auto i : c10::irange(size, device_count)) {
      device_dtbpool[i] = std::make_unique<CheckpointPool>();
      peak_allocated_memory[i] = 0;
      peak_reserved_memory[i] = 0;
      if_train_mode[i] = false;
      if_during_backward[i] = false;
    }
  }
}

#ifndef MEM_ORDER_ENABLE
void DTBCheckpointPool::auto_evict(int device) {                 /// TAG: multi mode evict entry with minimal eviction
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  auto pool = device_dtbpool[device].get();
  if (pool->has_memory_budget&&if_train_mode[device]) {
    int check_counts[8] = {0};
    bool if_eviction = false;
    size_t last_mem = current_memory(device);

    ///TODO: 更新当前依赖
#ifdef DEBUG_MODE
    if(record_dependcy&&current_memory(device) > pool->memory_budget){
      for (size_t i = 0; i < pool->aps.size(); i++) {
        auto ap_strong = pool->aps[i].lock();
        if (!ap_strong.defined()||ap_strong->ecn) {
          continue;
        } else {
          if (ap_strong->evictable()) {
            ap_strong->update_dependency();
          }
        }
      }
    }
#endif

    while (current_memory(device) > pool->memory_budget) {
      if_eviction = true;
#ifdef DEBUG_MODE
      if(record_mem_addr&&!current_if_any_evicted){
        at::DTRLogCounts("after computation, need evict.", ++evict_counts);
        pool_cur_mem_snapshot(device);  // after once computation, and out of budget
      }
      // current_if_any_evicted = true;
#endif
      pool->evict();
      check_counts[device]++;

      if(last_mem!=current_memory(device)){
        check_counts[device] = 0;
        last_mem = current_memory(device);
      }

      if(check_counts[device]>1000){
        throw std::runtime_error("Eviction progress has been trapped in dead loop, please check if any tensors ralated to operators(jit.fuse or custom kernel) isn't dealed with.");
      }
    }

#ifdef DEBUG_MODE
    if(record_mem_addr&&current_if_any_evicted){
      at::DTRLogCounts("after eviction.", evict_counts);
      pool_cur_mem_snapshot(device);  // after eviction
      current_if_any_evicted = false;
    }
    ///TODO: 清空依赖
    if(record_dependcy&&if_eviction){
      time_t current_time = std::chrono::system_clock::now();
      for (size_t i = 0; i < pool->aps.size(); i++) {
        auto ap_strong = pool->aps[i].lock();
        if (!ap_strong.defined()||ap_strong->ecn) {
          continue;
        } else {
          if (ap_strong->evictable()) {
            auto dep = ap_strong->get_dependency();
            // at::DTRLogCounts("ap dep", dep);
            at::DTRLogDepAndCost("ap dep", dep, ap_strong->cost(current_time));
          }
        }
      }
      at::DTRLogCounts("once check end", 999);
    }

    // if(if_eviction){
    //     // use_log_ = true;
    //   if(record_mem_addr && pool->exts.size()>10){
    //     /// 单独用这个，记录某次驱逐后的mem snapshot
    //     pool_cur_mem_snapshot(device);
    //     // if_rec = false;

    //     /// 单独用这个，记录每次驱逐后的内存变化
    //     // at::DTRLogMemAlloc(current_memory(), reserved_memory());
    //   // }
    //   }
    //   record_mem_addr = false;
    // }

#endif

  }
}
      
#else
void DTBCheckpointPool::auto_evict(int device) {                 /// TAG: multi mode evict entry with mem order
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  auto pool = device_dtbpool[device].get();
  if (pool->has_memory_budget) {
    size_t check_counts = 0;

    bool if_eviction = false;
    bool if_clear = false;
    while (current_memory(device) > pool->memory_budget) {
      if_eviction = true;
#ifdef DEBUG_MODE
      current_if_any_evicted = true;
#endif
      check_counts++;
      if(check_counts<10)
        pool->mem_first_evict(if_clear);
      else
        pool->exec_first_evict();

#ifdef DEBUG_MODE
      // if(check_counts>1000){
      //   std::cout << "Eviction progress has been trapped in dead loop\n";
      //   throw std::runtime_error("Eviction progress has been trapped in dead loop");
      // }
#endif

    }

#ifdef DEBUG_MODE
    if(check_counts>0){
      at::DTRLogCounts("search counts", check_counts);
      at::DTRLogMemAlloc(current_memory(), reserved_memory());
    }
    // if(if_eviction){
    //   if(record_mem_addr && pool->exts.size()>10){
    //     /// 单独用这个，记录某次驱逐后的mem snapshot
    //     pool_cur_mem_snapshot(device);
    //     /// 单独用这个，记录每次驱逐后的内存变化
    //     // at::DTRLogMemAlloc(current_memory(), reserved_memory());
    //   // }
    //   }
    // }
    //   record_mem_addr = false;
    if(record_mem_addr&&current_if_any_evicted){
      pool_cur_mem_snapshot(device);  // before computation
    }
#endif

  }
}
#endif

void DTBCheckpointPool::auto_evict(int device, size_t coming_bytes) {
  if(!device_id_check(device)) return;
  init_check();
  auto pool = device_dtbpool[device].get();
  if (pool->has_memory_budget&&if_train_mode[device]) {
    int check_counts[8] = {0};
    bool if_eviction = false;
    size_t last_mem = current_memory(device);
    while ((current_memory(device) + coming_bytes) > pool->memory_budget) {

      pool->evict();

      check_counts[device]++;
      if(last_mem!=current_memory(device)){
        check_counts[device] = 0;
        last_mem = current_memory(device);
      }
      if(check_counts[device]>1000){
        throw std::runtime_error("Eviction progress has been trapped in dead loop, please check if any tensors ralated to operators(jit.fuse or custom kernel) isn't dealed with.");
      }
    }

  }
}

void DTBCheckpointPool::force_evict(int device, int mode) {
  if(!device_id_check(device)) return;
  init_check();
  auto pool = device_dtbpool[device].get();
  pool->force_evict(mode);
}

void DTBCheckpointPool::add_ap(int device, const intrusive_ptr<AliasPool>& new_ap){
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  auto pool = device_dtbpool[device].get();
  pool->add(new_ap);
  // }else if(device==-1){
  //   for (const auto& pool : device_dtbpool) {
  //     pool->add(new_ap);
  //   }
  // }else{
  //   throw std::runtime_error("Invalid device was detected during ap inserting.");
  // }
}

void DTBCheckpointPool::add_ext(int device, const weak_intrusive_ptr<External>& new_ext) {
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  // if(likely(device>=0)){
  auto pool = device_dtbpool[device].get();
  pool->exts.push_back(new_ext);
  // }else if(device==-1){
  //   for (const auto& pool : device_dtbpool) {
  //     pool->exts.push_back(new_ext);
  //   }
  // }else{
  //   throw std::runtime_error("Invalid device was detected during exts inserting.");
  // }
}

void DTBCheckpointPool::add_into_keychain(int device, const weak& new_key, const weak& pre) {
  if(!device_id_check(device)) return;
  init_check();
  auto pool = device_dtbpool[device].get();
  pool->candidates.push_back(new_key);

  auto pre_node = StrongChainNode::make(pre);
  auto new_node = StrongChainNode::make(new_key);
  new_node->parent = pre_node;
  if(pool->chains.empty()){
    auto new_chain = ResidualChainRef::make(pre_node);
    new_chain->insert(new_node);
    pool->chains.push_back(new_chain);
  }else{
    bool is_find = false;
    for(const auto& chain: pool->chains){
      if(chain->in_chain(pre_node)){
        chain->insert(new_node);
        is_find = true;
        break;
      }
    }
    if(!is_find){
      auto new_chain = ResidualChainRef::make(pre_node);
      new_chain->insert(new_node);
      pool->chains.push_back(new_chain);
    }
  }

#ifdef DEBUG_MODE
  if(record_key_chain){
    if(pool->candidates.size()>0){
      auto chain_num = pool->chains.size();
      auto mem_num = pool->chains.back()->size();
      if(auto first_weak = pool->chains.back()->members[0]->value.lock())
        auto name = first_weak->counter_name();
      for(const auto& wc: pool->candidates){
        if(auto cell = wc.lock()){
          // at::DTRLogTensorInfo(cell->counter_name(), cell->pool->addr, cell->pool->memory, cell->get_degree(), 0, 0);
        }
      }
    }
    // record_key_chain = false;
  }
#endif
}

void DTBCheckpointPool::erase_ap(int device, uintptr_t addr){
  auto pool = device_dtbpool[device].get();
  pool->mem_ordered_aps.erase(addr);
}

void DTBCheckpointPool::toggle_sampling(bool if_sampling){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->set_sample_tensors(if_sampling);
    }
  }
}

void DTBCheckpointPool::toggle_ignore_small_tensors(bool if_ignore){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->set_ignore_small_tensors(if_ignore);
    }
  }
}

void DTBCheckpointPool::set_memory_budget(long budget){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->set_memory_budget(budget);
    }
  }
}

void DTBCheckpointPool::unset_memory_budget(){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->unset_memory_budget();
    }
  }
}

void DTBCheckpointPool::set_train_mode(bool flag){
  for (int i=0; i<if_train_mode.size(); i++) {
    if_train_mode[i] = flag;
  }
}

void DTBCheckpointPool::set_during_backward(bool flag){
  for (int i=0; i<if_during_backward.size(); i++) {
    if_during_backward[i] = flag;
  }
}

void DTBCheckpointPool::clear_checkpointpool(int device){
  if(device_dtbpool.empty()) return;          // exec without dtbpool  
  auto pool = device_dtbpool[device].get();
  if(pool->has_memory_budget){
    /// for debug, clear_exts will 
  #ifdef DEBUG_MODE
    // for (size_t i = 0; i < pool->aps.size(); i++) {
    //   auto ap_strong = pool->aps[i].lock();
    //   if (!ap_strong.defined()||ap_strong->ecn) {
    //     continue;
    //   } else {
    //     if (ap_strong->is_retain) {
    //       at::DTRLogCounts("ap tensors size", ap_strong->tensors.size());
    //       auto t = ap_strong->tensors.back().lock();
    //       at::DTRLogTensorInfo(t->counter_name(), ap_strong->addr, ap_strong->memory, 0, 0, 0);
    //       // for(const auto&t: ap_strong->tensors){
    //       //   auto cell = t.lock();
    //       //   if(cell->defined)
    //       //     at::DTRLogTensorInfo(cell->counter_name(), ap_strong->addr, ap_strong->memory, cell->get_degree(), 0, 0);
    //       // }
    //       ap_strong->unlock();
    //     }
    //   }
    // }
  #endif
    pool->clear_exts();
  }
}

void DTBCheckpointPool::pool_cur_mem_snapshot(int device){
//   time_t current_time = std::chrono::system_clock::now();
//   for(const auto& ex: device_dtbpool[device].get()->exts){ 
//     if(auto ref = ex.lock()){
//       if(ref->value->defined){
//         auto& remat = ref->value->remat;
//         size_t degree = 0;
//         double cost = 0;
//         size_t staleness = 0;
//         if(remat!=nullptr){ /// TODO: 存在没有remat的tensor 无源之水，从内存大小来看是一些用到的常量直接checkpoint
//           degree = (remat->inputs.size()) + (remat->outputs.size());
//           auto ap_strong = ref->value->pool; /// TODO: 这里会触发段错误，访问了野指针？
//           cost = ap_strong->cost(current_time);
//           staleness = (current_time-ap_strong->last_used_time).count();
//         }
//         // if(ref->value->pool->memory!=0)
// #ifdef DEBUG_MODE
//         at::DTRLogTensorInfo(ref->value->counter_name(), ref->value->pool->addr, ref->value->pool->memory, degree, cost, staleness);
// #endif
//       }
//     }
//   }
}

std::vector<std::pair<size_t, size_t>> DTBCheckpointPool::get_peak_memory(){
  std::vector<std::pair<size_t, size_t>> res;
  for (int i = 0; i<device_dtbpool.size(); i++) {
      res.push_back({peak_allocated_memory[i], peak_reserved_memory[i]});
  }
  return res;
}

DTBCheckpointPool dtb_pool;

struct BackendStaticInitializer{
  DTBCheckpointPool* parseEnvForBackend() {
    /// TODO: TO BE ADD STATIC ENV SETTING
    /* 
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != nullptr) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          if (kv[0] == "backend") {
            if (kv[1] == "cudaMallocAsync")
              return CudaMallocAsync::allocator();
            if (kv[1] == "native")
              return &Native::allocator;
          }
        }
      }
    }
    */
    return &dtb_pool;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    PoolManager.store(r);
  }
};

std::atomic<DTBCheckpointPool*> PoolManager;
BackendStaticInitializer backend_static_initializer;


void init(int device_count) {
  return get()->init(device_count);
}

void dtbPoolInitEntry() {
  const auto num_devices = c10::cuda::device_count_ensure_non_zero();
  c10::dtb::init(num_devices);
#ifdef DEBUG_MODE
  signal(SIGSEGV, signal_handler);
#endif
}

static c10::once_flag dtb_init;

void lazyInitDTB() {
  c10::call_once(dtb_init, [&] { dtbPoolInitEntry(); });
}

//use pool with call: auto* poolManager = getDTBPoolManager();
DTBCheckpointPool* getDTBPoolManager() {
  return c10::dtb::get();
}
  
}
}

