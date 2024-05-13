/*
  文件开头的宏定义是用于不同优化或者debug模式进行开关的
*/
#pragma once

#include <ATen/CheckpointTensorImpl.h>
// #include <c10/core/CheckpointTensorImpl.h>
// #include <ATen/Logger.h>
// #include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/dtb/DTBManager.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <string>
#include <random>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unistd.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>


namespace at {
  bool reserved_range = false;
  bool during_backward = false;
  // bool if_train_mode = false;
  // std::string INSTRUCTION = "INSTRUCTION";
  // std::string ANNOTATION = "ANNOTATION";
}

namespace c10 {
namespace dtb {

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Duration = Clock::duration;
using FinalTime = std::chrono::nanoseconds;
const time_t test_time_post = std::chrono::system_clock::now();
const time_t test_time_cur = std::chrono::system_clock::now();
auto test_dur = test_time_cur - test_time_post;


/* 这几项用于记录tensor内存分布特征 */
size_t memory_sum = 0;
size_t memory_max = 0;
size_t memory_count = 0;


long base_compute_time_ = 0;
long remat_compute_time_ = 0;
long search_time_ = 0;
long cost_time_ = 0;
bool use_log_ = false;
bool use_profile_ = false;
#ifdef DEBUG_MODE
bool record_er_counts = false;        // 驱逐&重物化次数
bool record_mem_addr = true;         // 是否记录内存地址
bool record_op_recs = false;          // 是否记录op历史
bool record_fragmentation = false;    // 记录碎片化和内存占用数据
bool record_lifecycle = false;        // 记录ap生命周期计数分布
bool record_ap_cost = false;          // 记录ap的cost分布
bool record_dependcy = false;
bool record_key_chain = false;
bool current_if_any_evicted = false;

std::atomic<size_t> evict_counts = 0;
std::atomic<size_t> tensor_evict_counts = 0;
std::atomic<size_t> remat_counts = 0;
std::atomic<size_t> cannot_evict_counts = 0;
std::atomic<size_t> destruct_counts = 0;
std::atomic<size_t> tensor_destruct_counts = 0;

void signal_handler(int sig) {
  constexpr const int REC_DEPTH = 50;
  void *array[REC_DEPTH];
  size_t size;
  char **strings;
  size_t i;

  // 获取当前的调用栈
  size = backtrace(array, REC_DEPTH);
  strings = backtrace_symbols(array, size);

  fprintf(stderr, "Error: signal %d:\n", sig);
  for (i = 0; i < size; i++) {
      fprintf(stderr, "%s\n", strings[i]);
  }

  free(strings);
  exit(1);
}

#endif

// bool reserved_range = false;
// bool during_backward = false;
// // bool if_train_mode = false;
// using at::reserved_range;
// using at::during_backward;


void reset_memory_stat() {
  memory_sum = 0;
  memory_max = 0;
  memory_count = 0;
}

struct PerfStats;

struct Timer {
  std::string name;
  Time start;
  Timer(std::string name, Time start) : name(name), start(start) {}
  Timer() {}
  ~Timer();
};

constexpr bool stats = true;

struct PerfStats {
  using TimerStats = std::tuple<std::string, Time, Time, Duration>;
  Time start;
  std::unordered_map<std::string, int> calls;
  std::vector<PerfStats::TimerStats> timers;

  PerfStats() : start(Clock::now()), calls(0), timers() {}

  /*Timer track(std::string name) {
    if (stats) {
    auto it = this->calls.find(name);
    if (it != this->calls.end()) {
      it->second += 1;
    } else {
      this->calls.insert({name, 0});
    }

    return Timer(name, Clock::now());
    }
    return Timer();
    }*/
  // for debug
  void track(const char*) { }

  ~PerfStats() {
    if (!stats) { return; }
    if(this->timers.size()>0){
      auto start = std::get<1>(this->timers[0]);
      auto now = Clock::now();
      std::cout << "All done. Here are some perf stats fresh off the preses." << std::endl;
      std::unordered_map<std::string, Duration> durations;

      Duration total = now - this->start;

      // For now simple strategy, count up all the time taken
      // by each "tagged call site".
      for (auto timer : timers) {
        auto name = std::get<0>(timer);
        Duration duration = std::get<3>(timer);
        auto it = durations.find(name);
        if (it != durations.end()) {
          it->second += duration;
        } else {
          durations.insert({name, duration});
        }
      }

      std::vector<std::pair<std::string, Duration>> data;

      // Convert the durations
      for (auto d : durations) {
        // auto duration = std::chrono::duration_cast<FinalTime>(d.second);
        data.push_back(d);
      }

      std::sort(data.begin(), data.end(),
      [](const std::pair<std::string, Duration> & a, const std::pair<std::string, Duration> & b) -> bool {
        return a.second > b.second;
      });

      for (auto d : data) {
        auto duration = std::chrono::duration_cast<FinalTime>(d.second);
        auto total_duration = std::chrono::duration_cast<FinalTime>(total);
        double percentage = ((double)duration.count())/((double)total_duration.count()) * 100;
        auto call_count = this->calls.find(d.first);
        TORCH_CHECK(call_count != this->calls.end());
        std::cout << "CallSite: " << d.first << " CallCount: " << call_count->second << " Cost: " << duration.count() << "ns" << " (%" << percentage << ")" << std::endl;
      }
    }
  }
};

PerfStats STATS = PerfStats();

Timer::~Timer() {
  Time now = Clock::now();
  Duration elapsed = now - start;
  PerfStats::TimerStats stats = { name , start, now, elapsed };
  STATS.timers.push_back(stats);
}

size_t memory(const Tensor& t) {
  if (! t.has_storage()) {
    return 0;
  }
  auto& storage = t.storage();
  size_t res = storage.nbytes();
  // these metrics are used for add aps
  memory_sum += res;
  memory_max = std::max(memory_max, res);
  memory_count += 1;
  return res;
}

uintptr_t get_addr(const Tensor& t) {
  if (!t.has_storage()) {
    return 0;
  }
  auto& storage = t.storage();
  auto res = storage.data_ptr().get();
  return reinterpret_cast<uintptr_t>(res);
}


/**
 * Methods of CheckpointPool
 * Implementation about eviction stragety
 * Management about AliasPools in the pool
*/
#pragma region CheckpointPool

CheckpointPool pool;  // cannot be extern
std::unordered_map<int64_t, duration_t> compute_cost_records;
std::unordered_map<int64_t, size_t> memory_cost_records;

void CheckpointPool::add(const intrusive_ptr<AliasPool>& p) {
  // ignore storage smaller than 1% average size
  if (p->memory > 0 && (memory_count == 0 || !ignore_small_tensors || p->memory >= 0.01 * double(memory_sum/memory_count))) {
    auto new_ap = weak_intrusive_ptr<AliasPool>(p);
    aps.push_back(new_ap);

#ifdef MEM_ORDER_ENABLE
    auto result = mem_ordered_aps.insert(std::make_pair(p->addr, new_ap));
    if (!result.second) {
        // 键已存在，更新其值
        result.first->second = new_ap;
    }
#endif
  }
}


#ifdef DEBUG_MODE
void log_cur_mem_statics(){ /// single mode use
#ifdef MULTI_MODE
    auto *pm = getDTBPoolManager();
    pm->pool_cur_mem_snapshot(0);
#else
    time_t current_time = std::chrono::system_clock::now();
    for(const auto& ex: pool.exts){
      if(auto ref = ex.lock()){
        if(ref->value->defined){
          auto& remat = ref->value->remat;
          size_t degree = 0;
          double cost = 0;
          if(remat!=nullptr){ /// TODO: 存在没有remat的tensor 无源之水，从内存大小来看是一些用到的常量直接checkpoint了，甚至有的常量读进来不用?
            degree = (remat->inputs.size()) + (remat->outputs.size());
            // auto ap_strong = ref->value->pool.get(); /// TODO: 这里会触发段错误，访问了野指针？
            // cost = ap_strong->cost(current_time);
          }
          // if(ref->value->pool->memory!=0)
          DTRLogTensorInfo(ref->value->counter_name(), ref->value->pool->addr, ref->value->pool->memory, degree, cost, 0);
        }
      }
        // DTRLogAddress(ref->value->counter_name(), ref->value->pool->addr, ref->value->pool->memory);
    }
    // DTRLogMemAlloc(current_memory(), reserved_memory());
#endif
}
#endif

// original dtr use this func. now use auto_evict of pm
void CheckpointPool::auto_evict() {
  STATS.track("CheckpointPool::auto_evict");
  constexpr float evict_mem_scale = 0.05;
  const size_t to_free_bytes = static_cast<size_t>(memory_budget * evict_mem_scale);
  if (has_memory_budget) {
    // if(current_memory() > memory_budget * (1-evict_mem_scale)){    /// TODO: 循环释放卡死
    //   std::cout<<"initiative trigger: " << current_memory();
    //   initiative_evict(to_free_bytes);
    //   std::cout<<" | " << current_memory() << "\n";
    // }
    // 使用cuda获取只能获取到reserved的情况，而pytorch存在自己的显存池，释放只是allocated部分发生了变化
    // 因此必须使用torch的CUDACachingAllocator获取显存情况
    int check_counts = 0;
    while (current_memory() > memory_budget) {
      // std::cout<<"passive trigger\n";
      evict();
      // evict(static_cast<size_t>(evict_mem_scale * memory_budget));
      // force_evict(0);
      check_counts++;
      if(check_counts>100){
        std::cout << "Eviction progress has been trapped in dead loop\n";
        throw std::runtime_error("Eviction progress has been trapped in dead loop");
      }
    }
#ifdef DEBUG_MODE
      // use_log_ = true;
      if(record_mem_addr){
        /// 单独用这个，记录某次驱逐后的mem snapshot
        log_cur_mem_statics();
        record_mem_addr = false;
        /// 单独用这个，记录每次驱逐后的内存变化
        // DTRLogMemAlloc(current_memory(), reserved_memory());
      // }
      }
#endif
  }
}

// bool peak_out = false;
void CheckpointPool::auto_evict(size_t size_bytes){
  STATS.track("CheckpointPool::auto_evict");
    if (has_memory_budget) {
    // 使用cuda获取只能获取到reserved的情况，而pytorch存在自己的显存池，释放只是allocated部分发生了变化
    // 因此必须使用torch的CUDACachingAllocator获取显存情况
    // auto peak = reserved_memory();
    // if(peak>memory_budget)
    //   peak_out = true;
    // //   throw std::runtime_error("Peak memory out.");
    // if(!peak_out){
    //   while ((reserved_memory() + size_bytes) > memory_budget) {
    //     evict();
    //   }
    // }else{
    while ((current_memory() + size_bytes) > memory_budget) {
      evict();
    }
    // }
  }
}

// for debug
void CheckpointPool::force_evict(int mode){
  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  // auto evict_from_idx = [&](size_t idx) {
  //                           auto ap_strong = aps[idx].lock();
  //                           TORCH_CHECK(ap_strong.defined());
  //                           ap_strong->evict(0);
  //                           remove_from_aps(idx);
  //                         };
  size_t i = 0;
  while(i<aps.size()){
    auto cannot_evict = [&]() {
                          // shrunk = true;
                          remove_from_aps(i);
                        };
    auto ap_strong = aps[i].lock();     // 此处的lock不是自定义的Lock // 这里发生过死锁
    if (!ap_strong.defined()) {
      cannot_evict();
    }
    else if (ap_strong->ecn) {    // 当aliaspool被驱逐后，会初始化其ecn，用于remat
#ifdef DEBUG_MODE
      if(record_er_counts){
        cannot_evict_counts += 1;
      }
#endif
      cannot_evict();
    }else{
#ifdef DEBUG_MODE
      if(record_er_counts){
        evict_counts += 1;
      }
#endif
      if(ap_strong->evictable()&&!ap_strong->is_retain){  // 不在保留区间内
        // evict_from_idx(i);
        TORCH_CHECK(ap_strong.defined());
        ap_strong->evict(0);
        remove_from_aps(i);
      }else{
        i++;
      }
    }
  }
}


void CheckpointPool::initiative_evict(size_t to_free_bytes){
  size_t have_freed_bytes = 0;
  constexpr size_t stride_idx = 1;
  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  auto evict_from_idx = [&](size_t idx) {
                            auto ap_strong = aps[idx].lock();
                            TORCH_CHECK(ap_strong.defined());
                            have_freed_bytes += ap_strong->memory;
                            ap_strong->evict(0);
                            remove_from_aps(idx);
                          };
  std::uniform_int_distribution<> distrib(1, 1 * std::max(1, static_cast<int>(std::sqrt(aps.size()))));
  // while(to_free_bytes > have_freed_bytes){
    for (size_t i = 0; i < aps.size();) {
      auto cannot_evict = [&]() {
                            // shrunk = true;
                            remove_from_aps(i);
                          };
      auto ap_strong = aps[i].lock();
      if (!ap_strong.defined()) {
        cannot_evict();
      }
      else if (ap_strong->ecn) {    // 当aliaspool被驱逐后，会初始化其ecn，用于remat
  #ifdef DEBUG_MODE
        if(record_er_counts){
          cannot_evict_counts += 1;
        }
  #endif
        cannot_evict();
      }
      else if (ap_strong->is_retain){
        i += stride_idx;
      }
      else{
  #ifdef DEBUG_MODE
        if(record_er_counts){
          evict_counts += 1;
        }
  #endif
        if(i%2==0){
          ap_strong->is_retain = true;
        }else{
          if (ap_strong->evictable()){
            evict_from_idx(i);
            // i--;
          }
        }
        i += stride_idx;
      }
      if(to_free_bytes <= have_freed_bytes)
        return;
    }
// }
}

void CheckpointPool::evict() {
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif 
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif
  STATS.track("CheckpointPool::evict");
  TORCH_CHECK(aps.size() > 0);
  // shrunk: either something has been evicted or the pools have gotten smaller
  bool shrunk = false;
  int evict_idx = -1;
  double evict_cost = INFINITY;
  time_t current_time = std::chrono::system_clock::now();

  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  std::uniform_int_distribution<> distrib(1, 1 * std::max(1, static_cast<int>(std::sqrt(aps.size()))));
  // sampling a random independent subset of all evictable tensors to find the cheapest tensor to evict.
  // 搜索策略，穷举搜索aps
  for (size_t i = 0; i < aps.size();) {
    auto cannot_evict = [&]() {
                          shrunk = true;
                          #ifdef DEBUG_MODE
                          if(record_er_counts){
                            cannot_evict_counts += 1;
                          }
                          #endif
                          remove_from_aps(i);
                        };
    auto ap_strong = aps[i].lock();
    if (!ap_strong.defined()) {
      cannot_evict();
    }
    else if (ap_strong->ecn) {
      cannot_evict();
    }
    else {
      if (ap_strong->evictable()) {
        double cost = ap_strong->cost(current_time);
      #ifdef DEBUG_MODE
        // if(record_ap_cost)
        //   DTRLogApCost("check cost", cost);
      #endif
        if (cost < evict_cost) {
          evict_cost = cost;
          evict_idx = i;
      #ifdef DEPENDENCY_CHECK
          ap_strong->update_dependency();
      #endif
        }
      }

      if (sample_tensors) {
        i += distrib(gen);
      } else {
        i += 1;
      }
    }
  }
  // 执行驱逐
  if (evict_idx == -1) {
    TORCH_CHECK(shrunk);
  } else {
#ifdef DEBUG_MODE
    if(record_er_counts){
      evict_counts += 1;
      // DTRLogEvictAPSEvents(evict_counts);
    }
#endif
    auto evict_from_idx = [&](size_t idx) {
                            auto ap_strong = aps[idx].lock();
                            TORCH_CHECK(ap_strong.defined());
                            ap_strong->evict(0);
                            remove_from_aps(evict_idx);
                          };
    auto ap_strong = aps[evict_idx].lock();
#ifdef DEPENDENCY_CHECK
    // ap_strong->update_dependency();
    auto cost_dep = ap_strong->get_dependency();
    if(cost_dep<dep_threshold){
      evict_from_idx(evict_idx);
    }else{
      threshold_touch_counts++;
      if(threshold_touch_counts%10==0&&dep_threshold<max_dep_threshold)
        dep_threshold *= 2;
    }
#else
    evict_from_idx(evict_idx);
#endif

#ifdef DEBUG_MODE
    if(record_ap_cost){
      // DTRLogApCost("evicted cost", evict_cost);
      DTRLogCounts("dependecy counts", ap_strong->dependency);
    }
    // if(record_mem_addr){  // TAG: mark evicted tensor for visualization
    //   for(const auto& wp: ap_strong->tensors){
    //     if(auto cp = wp.lock()){
    //       DTRLogTensorInfo(cp->counter_name(), cp->pool->addr, cp->pool->memory, 0, 0, -999);
    //     }
    //   }
    // }
#endif
  }
#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  search_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  search_time_ += (post - pre).count();
#endif
}

void CheckpointPool::exec_first_evict() {
  STATS.track("CheckpointPool::exec_first_evict");
  TORCH_CHECK(aps.size() > 0);
  // shrunk: either something has been evicted or the pools have gotten smaller
  bool shrunk = false;
  double evict_cost = INFINITY;
  auto best_it = mem_ordered_aps.begin();
  time_t current_time = std::chrono::system_clock::now();


  for (auto it = mem_ordered_aps.begin(); it != mem_ordered_aps.end();) {
    auto ap_strong = it->second.lock();
    if (!ap_strong.defined()||ap_strong->ecn) {
      it = mem_ordered_aps.erase(it);
    }
    else {
      if (ap_strong->evictable()) {
        double cost = ap_strong->cost(current_time);

        if (cost < evict_cost) {
          evict_cost = cost;
          best_it = it;
        }
      }
      it++;
      // if (sample_tensors) {
      //   i += distrib(gen);
      // } else {
      //   i += 1;
      // }
    }
  }
  // 执行驱逐
  auto ap_strong = best_it->second.lock();
  if(ap_strong->evictable()){
    ap_strong->evict(0);
    mem_ordered_aps.erase(best_it->first);
  }
}

/// @brief  TODO: bug 死锁
/// @param if_cleared 
void CheckpointPool::mem_first_evict(bool &if_cleared) {
  STATS.track("CheckpointPool::mem_first_evict");
  int evict_idx = -1;
  constexpr int search_box_size = 16;  // 相邻范围

  //   #ifdef DEBUG_MODE
      //   if(record_ap_cost)
      //     DTRLogApCost("check cost", cost);
    // #endif

  time_t current_time = std::chrono::system_clock::now();
  auto best_it = mem_ordered_aps.begin();
  double min_evict_cost = INFINITY;
  auto size_map = mem_ordered_aps.size();

  std::vector<uintptr_t> to_be_erase;
  if (size_map>search_box_size){

    for (auto it = mem_ordered_aps.begin(); std::distance(it, mem_ordered_aps.end()) >= search_box_size; it++) {
        auto end_it = std::next(it, search_box_size);
        double part_cost = 0;
        for (auto jt = it; jt != end_it; ++jt) {
          auto ap_strong = jt->second.lock();
          if (!ap_strong.defined()||ap_strong->ecn){
            to_be_erase.push_back(jt->first);
          }
          else if(!ap_strong->evictable()){  // 对不可驱逐的给一个比较大的惩罚 | 直接设为无穷会牺牲驱逐可能性
            part_cost += ap_strong->cost(current_time) * 100;
            // part_cost = INFINITY;
            // break;
          }else
            part_cost += ap_strong->cost(current_time); // 调用成员函数并累加返回值
        }

        if (part_cost < min_evict_cost) {
            min_evict_cost = part_cost;
            best_it = it;
        }

        // int i = 0;
        // while(i++ < search_box_size && std::distance(it, mem_ordered_aps.end()) >= search_box_size) it++;
    }

    // 删除具有最小代价的k个元素
    auto end_it = std::next(best_it, search_box_size);
    for(auto it = best_it; it != end_it; it++){
      auto ap_strong = it->second.lock();
      if(ap_strong.defined()&&ap_strong->evictable()){
        ap_strong->evict(0);
        to_be_erase.push_back(it->first);
      }
    }
  }else{
    for (auto it = mem_ordered_aps.begin(); it != mem_ordered_aps.end(); it++) {
      auto ap_strong = it->second.lock();
      if (!ap_strong.defined()||ap_strong->ecn){
        to_be_erase.push_back(it->first);
      }else if(ap_strong->evictable()){  // 对不可驱逐的给一个比较大的惩罚
        if (ap_strong->cost(current_time) < min_evict_cost) {
            min_evict_cost = ap_strong->cost(current_time);
            best_it = it;
        }
      }
    }
    auto ap_strong = best_it->second.lock();
    if(ap_strong.defined()&&ap_strong->evictable()){
      ap_strong->evict(0);
      to_be_erase.push_back(best_it->first);
    }
  }

  for(const auto& addr: to_be_erase)
    mem_ordered_aps.erase(addr);

}

void CheckpointPool::clear_exts(){
  candidates.clear();
  chains.clear();
#ifdef DEBUG_MODE
  // int count = 0, pool_count = 0;
  // std::map<uintptr_t, int> pool_rec;
#endif
  while (!exts.empty()) {
    if (auto e = exts.back().lock()) {
      // e->value->pin();  /// why pin and remat?
      if((e->value->pool->lock_count!=0||e->value->pool->external_count>0||e->value->pool->remat_count>0)&&e->value->defined){
#ifdef DEBUG_MODE
        // count++;
        // auto pool_ptr = reinterpret_cast<uintptr_t>((e->value->pool.get()));
        // auto it = pool_rec.find(pool_ptr);
        // int pool_id = 0;
        // if(it==pool_rec.end()){
        //   pool_rec[pool_ptr] = ++pool_count;
        //   pool_id = pool_count;
        // }else{
        //   pool_id = pool_rec[pool_ptr];
        // }
        // printf("exts size: %ld, size:%ld, external_count:%ld, is_weight:%d, pool_count:%d device_id:%d, have_remat:%d counts:%d\n", 
        //   exts.size(), e->value->pool->memory, e->value->pool->external_count, e->value->pool->if_weight ? 1 : 0, pool_id,
        //   e->value->pool->device_id, e->value->pool->head_remat ? 1 : 0, count);
#endif
        if(e->value->pool->external_count>1){     /// TODO: 这里仍然不是全明晰的，部分external_count释放后，会出现segmentation fault，目前是没有问题的
          e->value->pin();
          // e->release_resources();
          // e->value->pool->release_external();
        }
      }
    }
    exts.pop_back();
  }
}


CheckpointPool::CheckpointPool() { }

#pragma endregion

/**
 * Methods of AliasPool
 * Implementation about metrics updating and eviction behavior 
*/
#pragma region AliasPoolMethods

// An aliaspool cant register itself to the checkpointpool - you have to do it yourself.
AliasPool::AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, int device_id) :
  head_remat(head_remat),
  memory(memory),
  device_id(device_id),
  last_used_time(std::chrono::system_clock::now()) {
}

AliasPool::AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id) :
  head_remat(head_remat),
  memory(memory),
  addr(addr),
  device_id(device_id),
  last_used_time(std::chrono::system_clock::now()) {
}

AliasPool::AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory, uintptr_t addr, int device_id, bool if_w) :
  head_remat(head_remat),
  memory(memory),
  addr(addr),
  device_id(device_id),
  if_weight(if_w),
  last_used_time(std::chrono::system_clock::now()) {
}

void AliasPool::release_resources() {
  tensors.clear();
  neighbors.clear();
  head_remat.reset();
}

CheckpointInfo merge_cpi(CheckpointInfo l, CheckpointInfo r) {
  STATS.track("merge_cpi");
  return CheckpointInfo(l.compute_cost + r.compute_cost);
}

void AliasPool::evict(int mode) { // 0 - evict | 1 - deconstruct | 2 - Irreversible deconstruction
  STATS.track("AliasPool::evict");
  TORCH_CHECK(!ecn);
  if(mode!=2){
    ecn = head_remat->get_ecn();      /// 发生驱逐|可恢复释放行为，初始化ecn
    auto ecns = neighbor_ecn();
    for (const auto& necn : ecns) {
      merge<CheckpointInfo>(merge_cpi, ecn, necn);
    }
  }
  TORCH_CHECK(memory > 0);
  TORCH_CHECK(lock_count == 0);
  TORCH_CHECK(!is_evicted);
  is_evicted = true;
  for (const weak& w : tensors) {
    if (auto cell = w.lock()) {
#ifdef DEBUG_MODE
      if(record_er_counts){
        if(mode==0)
        {
          tensor_evict_counts += 1;
          // DTRLogEvictEvents(cell->counter_name(), tensor_evict_counts);
        }
        else
          tensor_destruct_counts += 1;
      }
#endif
      cell->evict();
    }
  }
  if(mode==1){  /// memory order use
    auto *pm = getDTBPoolManager();
    pm->erase_ap(device_id, addr);
  }
}

void AliasPool::unlock() {
  --lock_count;   // external == 0 , lock_count > 0 == 0
  /// improvement for life cycle
  /// because that staleness is harmful to eviction of remated tensor during backward progress, which should be released immediately
#ifndef ORIGINAL_DTR
  if(remat_count>0){
    unlock_remated();
  #ifdef DEBUG_MODE
    if(record_lifecycle){ // 这里记录的是重物化过程的情况
      pid_t pid = getpid();
      DTRLogLifeCycle(std::to_string(pid), external_count, lock_count, remat_count);
    }
  #endif
    if(remat_count == 0 && external_count == 0 && lock_count == 0 && retain_count == 0){
      if (memory > 0 && (!ecn) && head_remat) {
        evict(1);
      } 
      // else if (memory > 0 && head_remat==nullptr)
      //   evict(2);
    }
  }
  /**
   * 上面的重物化检查相当于提供了一个释放重物化张量的timing
   * 但实际上由于动态执行中会出现remat_count==0但lock_count>0导致无法回收的情况（错过了这个回收窗口）
   * 因此在反向过程中额外检查是否有释放的机会
  */
  if(during_backward){
    if(remat_count == 0 && external_count == 0 && lock_count == 0){
      if (memory > 0 && (!ecn) && head_remat) {
        evict(1);
      } 
    }
  }
#endif
}

void AliasPool::release_external() {
  --external_count;
  if (external_count == 0) {          /// TODO: 潜在bug，如果lock_count>0，此后这个aps会成为僵尸内存; 反向内存堆积的原因是否是因为这个？ 还是其他的引用计数
    if(if_weight) return;
    if (lock_count > 0) {
      return;
    }
    

    if (memory > 0 && (!ecn) && head_remat) {   /// [TAG] 对于无源之水无本之木，他们在这里就不会被释放了，包括所有模型权重和其他直接checkpoint的结果
#ifdef DEBUG_MODE
      // DTRLogDestructEvents();
      destruct_counts += 1;
#endif
      evict(1);
    } 
    // else if(memory > 0 && head_remat == nullptr){   /// 配合release_external_of_nosource_tensor才能释放这些张量
    //   evict(2);
    // }
  }
}

int AliasPool::update_dep_task(){
  auto& last_t = tensors.back();
  if(auto cell = last_t.lock()){
    return cell->precheck();
  }
  return 0;
}

void AliasPool::update_dependency() {
  dep_future = std::async(std::launch::async, [this]() { return this->update_dep_task(); });

  // for (size_t i = 0; i < tensors.size(); i++){
  //   {
  //     if(auto cell = tensors[i].lock()){
  //       if(cell->pool->dependency>0){
  //         dependency += cell->pool->dependency;
  //       }else{
  //         cell->precheck(dependency);
  //       }
  //     }
  //   }
  // }
}

double AliasPool::cost(time_t current_time) {
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif
  auto cpi = head_remat->get_cpi();
  auto ecns = neighbor_ecn();
  for (const auto& necn : ecns) {
    cpi = merge_cpi(cpi, get_t(necn));
  }
#ifdef DEPENDENCY_CHECK
  // 给依赖深的增加penalty
  auto ret = cpi.cost(memory, (current_time - last_used_time).count() * (1 + get_dependency() * 100));
#else
  auto ret = cpi.cost(memory, (current_time - last_used_time).count());
#endif
#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  cost_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  cost_time_ += (post - pre).count();
#endif
  return ret;
}

std::set<ecn_ptr> AliasPool::neighbor_ecn() {
  STATS.track("AliasPool::neighbor_ecn");
  std::set<ecn_ptr> ptr_set;
  int size = neighbors.size();
  for (size_t i = 0; i < size;) {
    if (auto cptc = neighbors[i].lock()) {
      if (cptc->pool->ecn) {
        ptr_set.insert(cptc->pool->ecn);
      }
      ++i;
    } else {
      neighbors[i] = neighbors[size - 1];
      size = size - 1;
    }
  }
  if (size < neighbors.size()) {
    neighbors.erase(neighbors.begin() + size);
  }
  return ptr_set;
}

void AliasPool::set_not_evicted(const intrusive_ptr<AliasPool>& self) {
  if (likely(is_evicted)) {
    STATS.track("AliasPool::set_not_evicted(inside)");
    is_evicted = false;
    if (ecn) {
      TORCH_CHECK(head_remat);
      auto cpi = get_t(ecn);
      update_t(ecn, CheckpointInfo(cpi.compute_cost - head_remat->compute_cost));
      ecn.reset();
    }

#ifdef MULTI_MODE
    auto *pm = getDTBPoolManager();
    pm->add_ap(device_id, self);
#else
    pool.add(self);
#endif
  }
}

#pragma endregion


/**
 * Methods of Rematerializer
 * Implementation about remat methods and several methods of packing and unpacking
*/
#pragma region RematerializerMethods

[[inline]]
Tensor uncheckpoint(const strong& input) {
  return input->get();
}

Tensors uncheckpoint(const strongs& inputs) {
  STATS.track("uncheckpoint");
  Tensors ret;
  ret.reserve(inputs.size());
  for (const strong& input : inputs) {
    // TAG: Remat entry
    ret.push_back(input->get());
  }
  return ret;
};

Tensors uncheckpoint_with_depth(const strongs& inputs, int& cumulative_num) {
  STATS.track("uncheckpoint");
  Tensors ret;
  ret.reserve(inputs.size());
  for (const strong& input : inputs) {
    // TAG: Remat entry
    /// TODO: 延长机制
    if(cumulative_num%42==0){
      input->pool->lock_retain();
    }
    ret.push_back(input->get(cumulative_num));
  }
  return ret;
};

Tensors try_checkpoint(Tensors& inputs) {
  STATS.track("try_checkpoint");
  Tensors ret;
  ret.reserve(inputs.size());
  for (auto& input : inputs) {
    ret.push_back(at::native::try_checkpoint(input));
  }
  return ret;
}

Rematerializer::Rematerializer(const Unsafe&,
                const rematerialize_function_t& func,
                const strongs& inputs,
                duration_t compute_cost)  :
  func(func),
  inputs(inputs),
  compute_cost(compute_cost) {
}

Rematerializer::Rematerializer(const Unsafe&,
                const rematerialize_function_t& func,
                const strongs& inputs,
                int64_t rid,
                duration_t compute_cost)  :
  func(func),
  inputs(inputs),
  rid(rid),
  compute_cost(compute_cost) {
}

void Rematerializer::release_resources() {
  func = rematerialize_function_t();
  inputs.clear();
  outputs.clear();
}


void Rematerializer::remat() {
#ifdef DEBUG_MODE
  if(record_er_counts){
    remat_counts += 1;
  }
  // if(remat_counts>1e5)
  //   throw std::runtime_error("Remat progress has been trapped in dead loop");
#endif
  // NOTE: author thinks that refactor using RAII for exception safety. however, RAII is not suitable for remat
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  Tensors ts = uncheckpoint(inputs);
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif

#ifdef MULTI_MODE
  auto device_id = static_cast<int>(ts[0].device().index());
  auto *pm = getDTBPoolManager();
#endif

#ifdef MINIMAL_EVICT_COST
  #ifdef MULTI_MODE
  pm->auto_evict(device_id, memory_cost_records[rid]);
  #else
  pool.auto_evict(memory_cost_records[rid]);
  #endif
#endif

  auto ret = func(ts);

#ifdef MINIMAL_EVICT
  #ifdef MULTI_MODE
  pm->auto_evict(device_id);
  #else
  pool.auto_evict();
  #endif
#endif

#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (auto output_cell = outputs[i].lock()) {
      output_cell->fill(ret[i]);
#ifndef ORIGINAL_DTR
      output_cell->pool->lock_remated();
#endif
    }
  }
  ecn.reset();
  for (const strong& s : inputs) {
    s->pool->unlock();
  }
}

void Rematerializer::remat(int& cumulative_num) {
#ifdef DEBUG_MODE
  if(record_er_counts){
    remat_counts += 1;
  }
  // if(remat_counts>1e5)
  //   throw std::runtime_error("Remat progress has been trapped in dead loop");
#endif
  // NOTE: author thinks that refactor using RAII for exception safety. however, RAII is not suitable for remat
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  cumulative_num++;
  Tensors ts = uncheckpoint_with_depth(inputs, cumulative_num);
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif

#ifdef MULTI_MODE
  auto device_id = static_cast<int>(ts[0].device().index());
  auto *pm = getDTBPoolManager();
#endif

#ifdef MINIMAL_EVICT_COST
  #ifdef MULTI_MODE
  pm->auto_evict(device_id, memory_cost_records[rid]);
  #else
  pool.auto_evict(memory_cost_records[rid]);
  #endif
#endif

  auto ret = func(ts);

#ifdef MINIMAL_EVICT
  #ifdef MULTI_MODE
  pm->auto_evict(device_id);
  #else
  pool.auto_evict();
  #endif
#endif

#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (auto output_cell = outputs[i].lock()) {
      output_cell->fill(ret[i]);
#ifndef ORIGINAL_DTR
      output_cell->pool->lock_remated();
#endif
    }
  }
  ecn.reset();
  for (const strong& s : inputs) {
    s->pool->unlock();
  }
}

ecn_ptr Rematerializer::get_ecn() {
  if (!ecn) {
    ecn = ecn_ptr::make(CheckpointInfo(compute_cost));
  }
  return ecn;
}

CheckpointInfo Rematerializer::get_cpi() {
  return CheckpointInfo(ecn ? duration_t(0) : compute_cost);
}

#pragma endregion


/**
 * Methods of CheckpointTensorCell && CheckpointTensorImpl
 * Basic but important methods about tensor's functionality
 * Important implementation about raw backend calling with pre-processing and post-processing
 * Register aliaspool and merge tensor sharing with the same ap
*/
#pragma region CheckpointTensorCellAndImpl

/**
 * original version of DTR use `const Tensor& t` as the argument type
 * and use it to construct a unique_ptr for manage the tensor storage.
 * However, const means cannot really implement the `move` of t, which is really
 * copy a Tensor to make a unique_ptr and do not affect the orignial tensor's storage.
 * 
 * But when using the `Tensor& t` as the argument type, which really `move` the t,
 * and manage t uniquely(almost manually), which may cause potential wild tensor and memory leak.
*/
void CheckpointTensorCell::fill(Tensor& t) {
  STATS.track("CheckpointTensorCell::fill");
  if (!(this->t)) {
    this->t = std::make_unique<Tensor>(std::move(t));
    pool->set_not_evicted(pool);                          /// TAG: 改变标志位，更新cost
    pool->set_addr(get_addr(t));
    if (!defined) {                                       /// 这里是将所有的属性拷贝一遍，虽然看起来毫无意义
      defined = true;
      is_undefined_tensor = !this->t->defined();
      key_set_ = this->t->key_set();
      // if (this->t->requires_grad()) {
      //   key_set_ = key_set_.add(DispatchKey::Autograd);
      // }
      dtype_ = this->t->dtype();
      if(this->t->defined())
        optional_device_ = this->t->device();
    }
  }
}

void CheckpointTensorCell::pin() {
  // get();         // [TAG] this is for debug to find out tensors unreleased
  /**
   * using at::strong = c10::intrusive_ptr<at::CheckpointTensorCell>
   * 这里释放的是重物化函数，然而整个系统中，只有remat中保留了strongs
   * 意味着释放重物化函数即可减少一定的strong计数
   * 
   * 另一处对strong有计数的是External，包裹在cpti里面，每个新的cpti会创建一个新的Externel with strong
   * cpti会对应着python阶段的tensor，因此当tensor释放时，对应的strong的计数会减一，对应pool的external_count也会减一
  */
  pool->head_remat.reset();
  remat.reset();
}

CheckpointTensorCell::CheckpointTensorCell(Tensor& t, const intrusive_ptr<AliasPool>& pool) : pool(pool) {
  fill(t);
}

CheckpointTensorCell::CheckpointTensorCell(Tensor& t,
                              const intrusive_ptr<AliasPool>& pool,
                              const intrusive_ptr<Rematerializer>& remat) :
  pool(pool), remat(remat) {
  fill(t);
}

void CheckpointTensorCell::evict() {
  TORCH_CHECK(remat);
  defined = false;
  t.reset();
}

void CheckpointTensorCell::release_resources() {
  defined = false;
  t.reset();
  pool.reset();
  remat.reset();
}

Tensor CheckpointTensorCell::get(){
  if (!t) {
      TORCH_CHECK(remat);
#ifdef DEBUG_MODE
      // if(record_er_counts)
      //   DTRLogRematEvents(counter_name(), 0);
#endif
      remat->remat();
    }
  defined = true;
  TORCH_CHECK(t);
  TORCH_CHECK(!t->key_set().has(DispatchKey::CheckpointTensorId));
  pool->last_used_time = std::chrono::system_clock::now();
  return *t;
}

Tensor CheckpointTensorCell::get(int& cumulative_num){  
  if (!t) {
    TORCH_CHECK(remat);
#ifdef DEBUG_MODE
    // if(record_er_counts)
    //   DTRLogRematEvents(counter_name(), 0);
#endif
    remat->remat(cumulative_num);
  }
  defined = true;
  TORCH_CHECK(t);
  TORCH_CHECK(!t->key_set().has(DispatchKey::CheckpointTensorId));
  pool->last_used_time = std::chrono::system_clock::now();
  return *t;
}

int CheckpointTensorCell::precheck(){  
  // 递归太慢
  // pool->set_dependency(dep);
  // if(dep>100) return;
  // remat->precheck(dep);

  int dependency = 0;
  std::queue<strong> sq;
  
  auto check_inputs = [&](const strongs& to_checks){
    for(auto& pret: to_checks){
      if(!pret->defined){ // 前驱不可用
        dependency++;
        sq.push(pret);
      }
    }
  };

  check_inputs(remat->inputs);
  
  while(!sq.empty()){
    const auto& check_ctc = sq.front();
    sq.pop();
    check_inputs(check_ctc->remat->inputs);
    if(dependency>dep_threshold)
      break;
  }

  return dependency;
  // pool->set_dependency(dependency);
}

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const{
  auto impl = c10::make_intrusive<CheckpointTensorImpl>(ref);
  TensorImpl::copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogCopy(impl->counter_name(), counter_name());
  }
#endif
  return impl;
}

// necessary in the process of autograd, use this func to copy and detach new tensor with cpti
c10::intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const{
  auto impl = c10::make_intrusive<CheckpointTensorImpl>(ref);
  TensorImpl::copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/std::move(version_counter),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogCopy(impl->counter_name(), counter_name());
  }
#endif
  return impl;
}


void CheckpointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  STATS.track("CheckpointTensorCell::shallow_copy_from");
  TORCH_CHECK(impl->key_set().has(DispatchKey::CheckpointTensorId));
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(impl.get());
  TORCH_CHECK(cpti != nullptr);
  ref->value = cpti->ref->value;
  if(unsafeGetTensorCell()->get().defined())
    set_sizes_and_strides(unsafeGetTensorCell()->get().sizes(), unsafeGetTensorCell()->get().strides());
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogCopyFrom(counter_name(), cpti->counter_name());
  }
#endif
}

#ifdef DEBUG_MODE
long CheckpointTensorCell::counter = 0;
#endif

bool is_alias(const Tensor& l, const Tensor& r) {
  return l.defined() && r.defined() && l.is_alias_of(r);
}

// return an index for alias.
// we dont care which one because they all lead to the same alias pool.
// return -1 for no alias.
// may god forgive my sin.
int get_alias(const Tensors& ts, const Tensor& t) {
  if (t.defined()) {
    for (size_t i = 0; i < ts.size(); ++i) {
      if (ts[i].defined() && t.is_alias_of(ts[i])) {
        return i;
      }
    }
  }
  return -1;
}

CheckpointTensorImpl* get_cpti(Tensor& t) {
  // auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  // if(cpti==nullptr){  
  //   t = at::native::checkpoint(t);
  //   auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  // }
  // return cpti;
  /// TODO: BUG 存在输入是在op内部创建的tensor，其并不是cpti
  // if(!t.is_checkpoint()){
  //   t = at::native::checkpoint(t);
  // }
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti;
}

Ref<intrusive_ptr<External>> cell_from_tensor(Tensor& t) {
  return get_cpti(t)->ref;
}


#ifdef DEBUG_MODE
struct OperationRecord {
  int64_t rid;
  std::string name;
  duration_t time;
  size_t mem;
  std::vector<string> inputs;
  std::vector<string> outputs;
  int device;
};
#endif

struct MakeRawResult {
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  duration_t time;
  intrusive_ptr<Rematerializer> rematerializer;
#ifdef DEBUG_MODE
  OperationRecord rec;
#endif
};

void add_neighbor(const strong& l, const strong& r) {
  l->pool->neighbors.push_back(weak(r));
  r->pool->neighbors.push_back(weak(l));
}

#ifdef DEBUG_MODE
#include <execinfo.h>
void printStackTrace() {
    const int maxFrames = 200; // Adjust the number of frames to print as needed
    void* callStack[maxFrames];
    int numFrames = backtrace(callStack, maxFrames);
    char** symbols = backtrace_symbols(callStack, numFrames);

    if (symbols != nullptr) {
        for (int i = 0; i < numFrames; ++i) {
            // Parse the symbol to extract file, function, and line information
            // The format is usually: "binary_name(function_name+offset) [file_path:line_number]"
            std::string symbol = symbols[i];

            // Find the opening and closing parentheses
            size_t openParenthesis = symbol.find("(");
            size_t closeParenthesis = symbol.find(")");

            if (openParenthesis != std::string::npos && closeParenthesis != std::string::npos) {
                // Extract the substring between parentheses
                std::string insideParentheses = symbol.substr(openParenthesis + 1, closeParenthesis - openParenthesis - 1);

                // Find the last occurrence of '+' to separate function name and offset
                size_t lastPlus = insideParentheses.rfind('+');
                if (lastPlus != std::string::npos) {
                    std::string function = insideParentheses.substr(0, lastPlus);
                    std::string offset = insideParentheses.substr(lastPlus + 1);

                    // Find the opening and closing brackets
                    size_t openBracket = symbol.find("[");
                    size_t closeBracket = symbol.find("]");

                    if (openBracket != std::string::npos && closeBracket != std::string::npos) {
                        std::string fileInfo = symbol.substr(openBracket + 1, closeBracket - openBracket - 1);

                        // Find the colon to separate file path and line number
                        size_t colon = fileInfo.find(":");
                        if (colon != std::string::npos) {
                            std::string filePath = fileInfo.substr(0, colon);
                            std::string lineNumber = fileInfo.substr(colon + 1);

                            std::cout << "Function: " << function << ", File: " << filePath << ", Line: " << lineNumber << std::endl;
                            continue;
                        }
                    }
                }
            }

            // Couldn't parse the symbol, just print it as is
            std::cout << symbols[i] << std::endl;
        }

        free(symbols);
    }
}
#endif

void if_tensor_add_key_chain(const weak& e, c10::dtb::DTBCheckpointPool *pm, int device){
  bool self_in = false, pre_in = false, post_in = false;
  weak pre = e;
  if(auto cell = e.lock()){
    if(C10_LIKELY(cell->get_degree()!=RESIDUAL_DEGREE)) return;
    self_in = true;
    if(C10_LIKELY(cell->remat)){
      for(const auto& it: cell->remat->inputs){
        if(it->get_degree()==RESIDUAL_DEGREE){
          pre_in = true;
          pre = weak(it);
          break;
        }
      }
      for(const auto& wot: cell->remat->outputs){
        if(auto ot = wot.lock()){
          if(ot->get_degree()==RESIDUAL_DEGREE){
            post_in = true;
            break;
          }
        }
      }
    }
  }
  if(self_in&&pre_in&&post_in){
    pm->add_into_keychain(device, e, pre);
  }
}

void if_tensor_add_key_chain(const strong& cell, c10::dtb::DTBCheckpointPool *pm, int device){
  bool self_in = false, pre_in = false, post_in = false;
  weak pre = weak(cell);
  if(C10_LIKELY(cell->get_degree()!=RESIDUAL_DEGREE)) return;
  self_in = true;
  if(C10_LIKELY(cell->remat)){
    for(const auto& it: cell->remat->inputs){
      if(it->get_degree()==RESIDUAL_DEGREE){
        pre_in = true;
        pre = weak(it);
        break;
      }
    }
    for(const auto& wot: cell->remat->outputs){ /// TODO: 这里其实是判断了自己，但是不进行这个判断会漏掉
      if(auto ot = wot.lock()){
        if(ot->get_degree()==RESIDUAL_DEGREE){
          post_in = true;
          break;
        }
      }
    }
  }
  if(self_in&&pre_in&&post_in){
    pm->add_into_keychain(device, weak(cell), pre);
  }
}

inline void release_external_of_nosource_tensor(const strong& s, const std::string& name){
  if(!s->pool->if_weight && s->pool->head_remat==nullptr && during_backward && name != "copy_"){    // [BUG]: copy_ is a complex bug for DTR runtime, cannot change orginal tensor easily
    // s->pool->release_external();                         /// BUG: this way will cause segmentation fault, probably related to free tensor needed
  }
}

// remat take a single vector of tensors,
// while there are two vector, one storing nonconstants and one storing constants.
// the constants are small and they will not be considered for eviction.
// however, we have to stitch the two vectors together to pass it in remat.
// the size_t in constants decide the location to stitch them in, while input_values fill in the rest.
MakeRawResult make_raw(const rematerialize_function_t& remat_f,
                       const strongs& inputs, const std::string& name) {
  STATS.track("make_raw");
  for (const strong& s : inputs) {                  // lock for unevictable
    s->pool->lock();
  }
#ifdef DEPTH_ENABLE
  int cumulative_num = 0;
  Tensors raw_inputs = uncheckpoint_with_depth(inputs, cumulative_num);
#else
  Tensors raw_inputs = uncheckpoint(inputs);        // cancel the monitor and get tensors
#endif
  auto device_id = static_cast<int>(raw_inputs[0].device().index());  /// do not influence device

  time_t pre = std::chrono::system_clock::now();
  auto raw_outputs = remat_f(raw_inputs);
  time_t post = std::chrono::system_clock::now();
  auto cur_compute_cost = post - pre;
  // auto cur_compute_cost = test_time_post - test_time_cur;

#ifdef MULTI_MODE
  auto* pm = getDTBPoolManager();
  pm->auto_evict(device_id);
#else
  pool.auto_evict();
#endif

#ifdef ORIGINAL_DTR
  base_compute_time_ += (post - pre).count();       // if sync?
  // c10::cuda::device_synchronize();
#endif
#ifdef TIMER_ENABLE
  base_compute_time_ += (post - pre).count();       // if sync?
#endif
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  weaks weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), remat_f, inputs, cur_compute_cost);

  for (Tensor& t : raw_outputs) {             // prepare checkpoint for raw_outputs
    intrusive_ptr<AliasPool> alias_pool;
    int alias = get_alias(raw_inputs, t);           // check if t is an alias of tensor in inputs
    if (alias == -1) {
      auto m = memory(t);
      auto addr = get_addr(t);
      // alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, device_id);
      alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, addr, device_id);     /// [TAG] AliasPool构造
      // if(reserved_range){     /// TAG: 保留区间内的aps保存
      //   alias_pool->is_retain = true;
      //   alias_pool->lock();    /// 一直保存了，需要主动释放
      // }
#ifdef MULTI_MODE
      pm->add_ap(device_id, alias_pool);
#else
      pool.add(alias_pool);     /// TAG: alaispool新增的唯一入口
#endif
    }
    else {
      alias_pool = inputs[alias]->pool;
      alias_pool->set_addr(get_addr(t));    // TODO: why org addr become a strange addr
      if (alias_pool->head_remat) {
        alias_pool->head_remat->compute_cost += cur_compute_cost;
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat); // bind external for t
#ifdef MULTI_MODE
    pm->add_ext(device_id, weak_intrusive_ptr<External>(e));
#else
    pool.exts.push_back(weak_intrusive_ptr<External>(e));
#endif
#ifdef DEGREE_CHAIN
    if(!during_backward&&pm->if_train_mode[device_id]){ /// change degree for outputs
      e->value->add_degree(inputs.size());
      // if_tensor_add_key_chain(weak(e->value), pm, device_id);  // output check cannot find residual
    }
#endif
    alias_pool->tensors.push_back(weak(e->value));                // same storage in one alias_pool
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }

  remat->outputs = weak_outputs;
  // make each pair of tensor which are not alias relation to be neighbor
  for (size_t i = 0; i < inputs.size(); ++i) {
#ifdef DEGREE_CHAIN
    if(!during_backward&&pm->if_train_mode[device_id]){   /// change degree for inputs
      inputs[i]->add_degree(outputs.size());
      if_tensor_add_key_chain(inputs[i], pm, device_id);
    }
#endif
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (!is_alias(raw_inputs[i], raw_outputs[j])) {
        add_neighbor(inputs[i], outputs[j]->value);
      }
    }
  }
  for (const strong& s : inputs) {
    s->pool->unlock();
    release_external_of_nosource_tensor(s, name);
  }

#ifdef DEBUG_MODE
  return {outputs, aliases, cur_compute_cost, remat, {{},{},{},{},{},{},{}}};
#else
  return {outputs, aliases, cur_compute_cost, remat};
#endif
}

template <class T>
inline void hash_combine(int64_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// 自定义的哈希函数, int64_t是IntArrayRef.vec()的默认类型
struct VectorHash {
    int64_t operator()(const std::vector<int64_t>& v) const {
        int64_t hash = 0;
        for (auto &num : v) {
            // 结合每个元素的哈希值，这里使用了位移和异或来混合哈希值  //  0×9e3779b9是 黄金分割数 乘 2^32次方 的16进制表示，32位无符号整数中它处在黄金分割的位置
            hash_combine(hash, num);
        }
        return hash;
    }
};

// 添加缓存机制来避免重复哈希字符串
std::unordered_map<std::string, int64_t> functionNameHashCache;

inline int64_t generateUniqueHash(const std::string& functionName, const Tensors &inputs) {
    int64_t hash = 0;

    // 从缓存中获取或计算函数名的哈希值
    auto it = functionNameHashCache.find(functionName);
    if (it == functionNameHashCache.end()) {
        int64_t fnHash = std::hash<std::string>{}(functionName) + 0x9e3779b9;
        functionNameHashCache[functionName] = fnHash;
        hash ^= fnHash;
    } else {
        hash ^= it->second;
    }

    // 为输入尺寸生成哈希值并合并
    VectorHash vectorHasher;
    for(auto &input: inputs){
        hash_combine(hash, vectorHasher(input.sizes().vec()));
    }

    return hash;
}


// func name and size for cost model
MakeRawResult make_raw_rec(const rematerialize_function_t& remat_f,
                       const strongs& inputs, const std::string& name) {
  STATS.track("make_raw with hash");

  for (const strong& s : inputs) {                  // lock for unevictable
    s->pool->lock();
  }
#ifdef DEPTH_ENABLE
  int cumulative_num = 0;
  Tensors raw_inputs = uncheckpoint_with_depth(inputs, cumulative_num);
#else
  Tensors raw_inputs = uncheckpoint(inputs);        // cancel the monitor and get tensors
#endif
  Tensors raw_outputs;
  auto device_id = static_cast<int>(raw_inputs[0].device().index());

  auto rid = generateUniqueHash(name, raw_inputs);
  auto search_res = compute_cost_records.find(rid);
  bool have_record = false;
  duration_t cur_compute_cost;
  size_t cur_mem_cost;
  if(search_res != compute_cost_records.end()){ // 有记录
    have_record = true;
  }

#ifdef DEPTH_ENABLE
  if(cumulative_num>0){
    DTRLogCalculativeRematsRecords(rid, name, cumulative_num);
  }
#endif

#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
#endif
  if(have_record){
    cur_mem_cost = memory_cost_records[rid];
#ifdef MULTI_MODE
    pm->auto_evict(device_id, cur_mem_cost);
#else
    pool.auto_evict(cur_mem_cost);
#endif
    cur_compute_cost = compute_cost_records[rid];
    raw_outputs = remat_f(raw_inputs);
  }else{
    size_t pre_mem = current_memory();
    time_t pre_time = std::chrono::system_clock::now();
    raw_outputs = remat_f(raw_inputs);
    time_t post_time = std::chrono::system_clock::now();
    size_t post_mem = current_memory();
    cur_compute_cost = post_time - pre_time;
    // insert new record item
    compute_cost_records[rid] = cur_compute_cost;
    cur_mem_cost = post_mem - pre_mem;
    memory_cost_records[rid] = cur_mem_cost;
#ifdef MULTI_MODE
    pm->auto_evict(device_id);
#else
    pool.auto_evict();
#endif
  }
  
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  weaks weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), remat_f, inputs, rid, cur_compute_cost);

  for (Tensor& t : raw_outputs) {             // prepare checkpoint for raw_outputs
    intrusive_ptr<AliasPool> alias_pool;
    int alias = get_alias(raw_inputs, t);           // if t is an alias of tensor in inputs?
    if (alias == -1) {
      auto m = memory(t);
      // alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, device_id);
      alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, get_addr(t), device_id);    /// [TAG] AliasPool构造
#ifdef MULTI_MODE
      pm->add_ap(device_id, alias_pool);
#else
      pool.add(alias_pool);     /// TAG: alaispool新增的唯一入口
#endif
    }
    else {
      alias_pool = inputs[alias]->pool;
      if (alias_pool->head_remat) {
        alias_pool->head_remat->compute_cost += (cur_compute_cost);
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat); // bind external for t
#ifdef MULTI_MODE
    pm->add_ext(device_id, weak_intrusive_ptr<External>(e));
#else
    pool.exts.push_back(weak_intrusive_ptr<External>(e));
#endif
    alias_pool->tensors.push_back(weak(e->value));                // same storage in one alias_pool
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }

  remat->outputs = weak_outputs;
  // make each pair of tensor which are not alias relation to be neighbor
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (!is_alias(raw_inputs[i], raw_outputs[j])) {
        add_neighbor(inputs[i], outputs[j]->value);
      }
    }
  }
  for (const strong& s : inputs) {
    s->pool->unlock();
    // if(!s->pool->if_weight && s->pool->head_remat==nullptr && during_backward && name != "copy_"){    // [BUG]: copy_ is a complex bug for DTR runtime
    //   // printf("[UNLOCK] %s %ld %ld %ld %s %s\n", s->counter_name().c_str(), s->pool->external_count, s->pool->tensors.size(),
    //   //       s->pool->memory, during_backward ? "in backward" : "in forward", name.c_str());
    //   s->pool->release_external();
    // }
    release_external_of_nosource_tensor(s, name);
  }
#ifdef DEBUG_MODE
  OperationRecord rec = {rid, name, cur_compute_cost, cur_mem_cost, {}, {}, device_id};
  return {outputs, aliases, cur_compute_cost, remat, rec};
#else
  return {outputs, aliases, cur_compute_cost, remat};
#endif
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

/**
 * 广泛使用在Checkpoint后端
*/
Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   Tensors& inputs) {
  STATS.track("CheckPointTensorImpl::make");
  Tensors checkpointed_inputs = try_checkpoint(inputs); // make intrusive_ptr for inputs
  auto input_size = checkpointed_inputs.size();

  strongs input_values;
  input_values.reserve(input_size);

  std::vector<std::string> args;
  args.reserve(input_size);

  for (const Tensor& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    // if(cpti->unsafeGetTensorCell()->pool->external_count>1){
    //   if(record_lifecycle){ // 记录关注的tensor
    //     DTRLogLifeCycle(name, cpti->unsafeGetTensorCell()->pool->external_count, cpti->unsafeGetTensorCell()->pool->lock_count, cpti->unsafeGetTensorCell()->pool->remat_count);
    //   }
    // }
    
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }

#ifdef MINIMAL_EVICT
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  Tensors tensors;
  tensors.reserve(ret.outputs.size());

  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
  }

#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (const auto& t: tensors) {
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      output_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  // if (use_log_) {
  //   std::vector<std::string> res;
  //   res.reserve(ret.outputs.size());

  //   for (const auto& tensor : tensors) {
  //     res.push_back(get_cpti(tensor)->counter_name());
  //   }

  //   DTRLogCall(res, name, args, from_time(ret.time));
  //   for (size_t i = 0; i < tensors.size(); ++i) {
  //     Tensor t = tensors[i];
  //     auto cpti = get_cpti(t);
  //     DTRLogMemory(cpti->counter_name(), cpti->unsafeGetTensorCell()->memory());
  //     DTRLogAlias(cpti->counter_name(), ret.aliases[i]);
  //     // DTRLogAlias(cpti->counter_name(), ret.aliases[i], cpti->unsafeGetTensorCell()->pool->tensors.size());
  //   }
  // }
#endif

  return tensors;
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   Tensors&& inputs) {
  STATS.track("CheckPointTensorImpl::make");
  Tensors checkpointed_inputs = try_checkpoint(inputs); // make intrusive_ptr for inputs
  auto input_size = checkpointed_inputs.size();

  strongs input_values;
  input_values.reserve(input_size);

  std::vector<std::string> args;
  args.reserve(input_size);

  for (const Tensor& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    // if(cpti->unsafeGetTensorCell()->pool->external_count>1){
    //   if(record_lifecycle){ // 记录关注的tensor
    //     DTRLogLifeCycle(name, cpti->unsafeGetTensorCell()->pool->external_count, cpti->unsafeGetTensorCell()->pool->lock_count, cpti->unsafeGetTensorCell()->pool->remat_count);
    //   }
    // }
    
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }

#ifdef MINIMAL_EVICT
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  Tensors tensors;
  tensors.reserve(ret.outputs.size());

  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
  }

#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (const auto& t: tensors) {
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      output_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  // if (use_log_) {
  //   std::vector<std::string> res;
  //   res.reserve(ret.outputs.size());

  //   for (const auto& tensor : tensors) {
  //     res.push_back(get_cpti(tensor)->counter_name());
  //   }

  //   DTRLogCall(res, name, args, from_time(ret.time));
  //   for (size_t i = 0; i < tensors.size(); ++i) {
  //     Tensor t = tensors[i];
  //     auto cpti = get_cpti(t);
  //     DTRLogMemory(cpti->counter_name(), cpti->unsafeGetTensorCell()->memory());
  //     DTRLogAlias(cpti->counter_name(), ret.aliases[i]);
  //     // DTRLogAlias(cpti->counter_name(), ret.aliases[i], cpti->unsafeGetTensorCell()->pool->tensors.size());
  //   }
  // }
#endif

  return tensors;
}

// TODO: check that mutated value does not have alias.
void CheckpointTensorImpl::mutate(const std::string& name,
                                  const mutate_function_t& mutate,
                                  Tensors& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                  //  new_input_values[idx] = t[idx].clone();    /// TODO: 绕开clone
                   new_input_values[idx] = t[idx];
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  strongs input_values;
  std::vector<std::string> args;
  // 拿到tensor本体
  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }
#ifdef MINIMAL_EVICT
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  const auto& modified = ret.outputs;
  for (size_t idx: mutate_idx) {                              /// TODO: 可能存在inputs中不为cpti的tensor，但受限于语法无法直接修改
    // if(C10_UNLIKELY(!native::is_checkpoint(inputs[idx])))
    //   inputs[idx] = native::checkpoint(inputs[idx]);
    if(!inputs[idx].is_checkpoint()){
      inputs[idx] = at::native::checkpoint(inputs[idx]);
    }
    cell_from_tensor(inputs[idx])->value = modified[idx];
  }
#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (size_t idx: mutate_idx) {
      output_ids.push_back(cell_from_tensor(inputs[idx])->value->value->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  if (use_log_) {
    DTRLogMutate(name, args, mutate_idx, from_time(ret.time));
  }
#endif
}

void CheckpointTensorImpl::mutate(const std::string& name,
                                  const mutate_function_t& mutate,
                                  Tensors&& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                  //  new_input_values[idx] = t[idx].clone();    /// TODO: 绕开clone
                   new_input_values[idx] = t[idx];
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  strongs input_values;
  std::vector<std::string> args;
  // 拿到tensor本体
  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }
#ifdef MINIMAL_EVICT
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  const auto& modified = ret.outputs;
  /**
   * 需要说明的是，这里不论是传入右值引用还是左值引用，都无法对原本的Tensor进行修改
   * 这里的处理只是让程序执行时不存在bug，并不能达到真正的目的——让非cpti的Tensor变为cpti
   * 在checkpoint.cpp中封装的函数，传入值存在const
  */
  for (size_t idx: mutate_idx) {                              /// TODO: 可能存在inputs中不为cpti的tensor，但受限于语法无法直接修改
    if(C10_UNLIKELY(!at::native::is_checkpoint(inputs[idx])))
      inputs[idx] = at::native::checkpoint(inputs[idx]);
    cell_from_tensor(inputs[idx])->value = modified[idx];
  }
#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (size_t idx: mutate_idx) {
      output_ids.push_back(cell_from_tensor(inputs[idx])->value->value->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  if (use_log_) {
    DTRLogMutate(name, args, mutate_idx, from_time(ret.time));
  }
#endif
}

void CheckpointTensorImpl::release_resources() {
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogRelease(counter_name());
  }
#endif
  ref.reset();
}

CheckpointTensorImpl::CheckpointTensorImpl(Tensor& t, bool if_weight) : CheckpointTensorImpl(c10::make_intrusive<External>(t, if_weight)) {
  // set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
  // set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);
  if(unsafeGetTensorCell()->get().defined())
  {
    set_sizes_and_strides(unsafeGetTensorCell()->get().sizes(), unsafeGetTensorCell()->get().strides());
    /// original DTR do not add directly transfered tensor into it's pool
    /// this is for the Fine grained memory management
    // unsafeGetTensorCell()->pool->tensors.push_back(weak(unsafeGetTensorCell()));         /// TODO: this should distinguish if outer tensor or inner tensor, otherwise double free
  }
#ifdef MULTI_MODE
  auto device_id = C10_LIKELY(t.defined()) ? static_cast<int>(t.device().index()) : -1;      /// CPU data possible or undefiend tensor
  auto *pm = getDTBPoolManager();
  pm->add_ext(device_id, weak_intrusive_ptr<External>(ref->value));
#else
  pool.exts.push_back(weak_intrusive_ptr<External>(ref->value));
#endif
}

#pragma endregion



}
}


namespace at {


// using c10::dtb::STATS;
using c10::dtb::getDTBPoolManager;
using c10::dtb::use_log_;
using c10::dtb::base_compute_time_;
using c10::dtb::remat_compute_time_;
using c10::dtb::search_time_;
using c10::dtb::cost_time_;
using c10::dtb::use_profile_;

using c10::dtb::base_compute_time_;
using c10::dtb::remat_compute_time_;
using c10::dtb::search_time_;
using c10::dtb::cost_time_;
using c10::dtb::use_log_;
using c10::dtb::use_profile_;
#ifdef DEBUG_MODE
using c10::dtb::record_er_counts;        // 驱逐&重物化次数
using c10::dtb::record_mem_addr;         // 是否记录内存地址
using c10::dtb::record_op_recs;          // 是否记录op历史
using c10::dtb::record_fragmentation;    // 记录碎片化和内存占用数据
using c10::dtb::record_lifecycle;        // 记录ap生命周期计数分布
using c10::dtb::record_ap_cost;          // 记录ap的cost分布
using c10::dtb::record_dependcy;
using c10::dtb::record_key_chain;
using c10::dtb::current_if_any_evicted;

using c10::dtb::evict_counts;
using c10::dtb::tensor_evict_counts;
using c10::dtb::remat_counts;
using c10::dtb::cannot_evict_counts;
using c10::dtb::destruct_counts;
using c10::dtb::tensor_destruct_counts;

#endif
/**
 * Interface expose to at native
 * Kinds of log methods
 * Initiative methods in Aten
*/
#pragma region InterfaceForAtenNative

namespace native {

/**
 * This function is used to transfer the original tensor created by torch
 * and attain the ability to manage their lifecyle.
 * 
 * @param t: original tensor
 * @param if_weight: if this tensor is a weight tensor or you do not wish it to be destroyed
 * 
 * @return res: tensor returned is a "empty" tensor without storage, which is moved into cpti
 * 
 * 
 * @skip
 * ---------------------------------------some debug notes-------------------------------------
 * !TODO: 潜在问题 引入张量t有undefined的情况，此种情况cpti构造出来的tensor却不是undefined的了，可以额外标记tensor是undefined
 * 需要规避空tensor带来的可能问题，调用empty tensor的成员方法会报错
 * [TAG] if_weight的标记操作只是保证了生命周期的合理性，即权重不释放，非权重的无源tensor释放
*/
Tensor checkpoint(Tensor& t, bool if_weight) {
  // STATS.track("checkpoint");
  // if(!t.defined())
  //   return Tensor(nullptr);
  // auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t);   // 调用了Ref<intrusive_ptr<External>> External CheckpointTensorCell的相关构造函数
  auto cpti = c10::make_intrusive<CheckpointTensorImpl>(t, if_weight);      // cpti->ref->value->value->t 是包裹的unique_ptr<Tensor> unsafeGetTensorCell()
#ifdef DEBUG_MODE
  if (use_log_) {
    c10::dtb::DTRLogConstant(cpti->counter_name());
    c10::dtb::DTRLogMemory(std::to_string(cpti->unsafeGetTensorCell()->pool->if_weight)+"-"+std::to_string(if_weight), cpti->unsafeGetTensorCell()->memory());
  }
#endif
  auto res = Tensor(cpti);
  return res;
}

Tensor uncheckpoint(const Tensor& t) {
  // STATS.track("uncheckpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti->unsafeGetTensorCell()->get();
}

/**
 * 支持上层操作的下层实现接口
*/
void pin(Tensor& t) {
  // STATS.track("pin");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  cpti->unsafeGetTensorCell()->pin();
}

Tensor decheckpoint(const Tensor& t) {
  // STATS.track("decheckpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  if(cpti){
    auto res = cpti->unsafeGetTensorCell()->get();
    return res.detach();
  }else
    return t;
  // return cpti ? cpti->unsafeGetTensorCell()->get() : t;
}

void cpti_decrease(const Tensor& t) {
  // STATS.track("decheckpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  if(cpti) cpti->release_resources();
}

bool is_checkpoint(const Tensor& t) {
  // STATS.track("is_checkpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

Tensor try_checkpoint(Tensor& t) {
  // STATS.track("try_checkpiont");
  // if(t.key_set().has(DispatchKey::Checkpoint)&&!is_checkpoint(t)){   /// 这行代码不一定有用 但debug验证的代价较大
  //   // t.key_set() = t.key_set().remove(DispatchKey::Checkpoint);    // 返回值是keyset，但没有set函数 所以是没有用的
  //   annotate_log("trigger");
  //   return(checkpoint(t.decheckpoint()));
  // }
  return is_checkpoint(t) ? t : checkpoint(t);
}

void new_log(std::string str) {
  c10::dtb::DTRLogger::logger().out = std::ofstream(c10::dtb::DTRLogger::logger().get_filename(str));
}

void annotate_log(c10::string_view str_) {
  // auto str = std::string(str_);
  // if (!use_log_) { return; }
  // if (c10::dtb::log_json) {
  //   c10::dtb::json j;
  //   j["INSTRUCTION"] = "ANNOTATE";
  //   j["ANNOTATION"] = str;
  //   c10::dtb::DTRLogger::logger().log(j.dump());
  // } else {
  //   c10::dtb::DTRLogger::logger().log("# " + str);
  // }
}

void toggle_log(bool b) {
  use_log_ = b;
}

void clear_checkpointpool(long device) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->clear_checkpointpool(device);
#else
  while (likely(!pool.exts.empty())) {
    if (auto e = pool.exts.back().lock()) {
      e->value->pin();
    }
    pool.exts.pop_back();
  }
#endif
}

void init_dtb_manager(){
#ifdef MULTI_MODE
  c10::dtb::lazyInitDTB();
#endif
}

void unset_memory_budget() {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->unset_memory_budget();
#else
  pool.has_memory_budget = false;
#endif
}

void set_memory_budget(long budget) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->set_memory_budget(budget);
#else
  pool.memory_budget = budget;
  pool.has_memory_budget = true;
#endif
}

void set_reserved(){
  reserved_range = true;
}

void unset_reserved(){
  reserved_range = false;
}

void set_backward_flag(){
// #ifdef MULTI_MODE
//   auto *pm = getDTBPoolManager();
//   pm->set_during_backward(true);
// #else
  during_backward = true;
// #endif
  // printf("SET_BACKWARD_FALG TRIGGER\n");
}

void unset_backward_flag(){
// #ifdef MULTI_MODE
//   auto *pm = getDTBPoolManager();
//   pm->set_during_backward(false);
// #else
  during_backward = false;
// #endif
  // printf("UNSET_BACKWARD_FALG TRIGGER\n");
}

void mark_train(bool flag){
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->set_train_mode(flag);
#else
  if_train_mode = flag;
#endif
}

/// TODO: use dtb, useless here
void force_evict(long mode){
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->force_evict(0, mode);
#else
  pool.force_evict(mode);
#endif
}

void log_dtr_statics(){
#ifdef DEBUG_MODE
#ifdef MULTI_MODE
  if(record_fragmentation){
    auto *pm = getDTBPoolManager();
    int did = 0;
    for(const auto& mem_info: pm->get_peak_memory()){
      std::stringstream log_str;
      log_str << "device-" << did << " peak allocated memory";
      c10::dtb::DTRLogCounts(log_str.str(), mem_info.first);
      log_str.clear();
      log_str << "device-" << did << " peak reserved memory";
      c10::dtb::DTRLogCounts(log_str.str(), mem_info.second);
      log_str.clear();
      log_str << "device-" << did << " fragmentation ratio";
      c10::dtb::DTRLogCounts(log_str.str(), static_cast<double>(mem_info.first) / static_cast<double>(mem_info.second));
      did++;
    }
  }
#endif
  if(record_er_counts){
    // DTRLogCounts("memory budget", memo);
    c10::dtb::DTRLogCounts("evict counts", evict_counts);
    c10::dtb::DTRLogCounts("evict tensor counts", tensor_evict_counts);
    c10::dtb::DTRLogCounts("cannot evict counts", cannot_evict_counts);
    c10::dtb::DTRLogCounts("destruct counts", destruct_counts);
    c10::dtb::DTRLogCounts("destruct tensor counts", tensor_destruct_counts);
    c10::dtb::DTRLogCounts("remat counts", remat_counts);
  }
#endif
}

/// TODO: use dtb
void toggle_sampling(bool sample) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->toggle_sampling(sample);
#else
  pool.sample_tensors = sample;
#endif
}

/// TODO: as pool member function
void toggle_ignore_small_tensors(bool ignore) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->toggle_ignore_small_tensors(ignore);
#else
  pool.ignore_small_tensors = ignore;
#endif
}

void reset_profile() {
  base_compute_time_ = 0;
  remat_compute_time_ = 0;
  search_time_ = 0;
  cost_time_ = 0;
}

void toggle_profile(bool profile) {
  use_profile_ = profile;
}

long compute_time() {
  return base_compute_time() + remat_compute_time();
}

long cost_time() {
  return cost_time_;
}

long search_time() {
  return search_time_;
}

long remat_compute_time() {
  return remat_compute_time_;
}

long base_compute_time() {
  return base_compute_time_;
}

long loop_time() {
  return search_time() - cost_time();
}

}
#pragma endregion
}

