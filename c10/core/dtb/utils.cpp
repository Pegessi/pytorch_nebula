#pragma once

#include <c10/core/dtb/utils.h>
#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/core/dtb/CheckpointTensorImpl.h>
#include <c10/cuda/dtb/DTBManager.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>

#define TORCH_CHECK(a, ...)   // replace original TORCH_CHECK  profile mode

namespace at {
  bool reserved_range = false;
  bool during_backward = false;
  // bool if_train_mode = false;
  // std::string INSTRUCTION = "INSTRUCTION";
  // std::string ANNOTATION = "ANNOTATION";
}

namespace c10 {
namespace dtb {

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
std::unordered_map<int64_t, duration_t> compute_cost_records;
std::unordered_map<int64_t, size_t> memory_cost_records;
size_t memory_budget = 85899345920;  
bool store_in_special_pool[8] = {false};

#ifdef DEBUG_MODE
bool record_er_counts = false;        // 驱逐&重物化次数
bool record_mem_addr = false;         // 是否记录内存地址
bool record_op_recs = false;          // 是否记录op历史
bool record_fragmentation = false;    // 记录碎片化和内存占用数据
bool record_lifecycle = false;        // 记录ap生命周期计数分布
bool record_ap_cost = false;          // 记录ap的cost分布
bool record_dependcy = false;
bool record_key_chain = false;
bool current_if_any_evicted = false;
bool trace_register_and_release = false;
bool trace_evicted_tensor = false;

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


Timer::Timer(std::string name, Time start) : name(name), start(start) {}

PerfStats::PerfStats() : start(Clock::now()), calls(0), timers() {}

PerfStats::~PerfStats() {
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

PerfStats STATS = PerfStats();

Timer::~Timer() {
  Time now = Clock::now();
  Duration elapsed = now - start;
  PerfStats::TimerStats stats = { name , start, now, elapsed };
  STATS.timers.push_back(stats);
}

size_t memory(const Tensor& t) {
  if (!t.has_storage()) {
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

void set_global_memory_budget(size_t budget){
  memory_budget = budget;
}

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
    // if(cumulative_num%42==0){
    //   input->pool->lock_retain();
    // }
    ret.push_back(input->get(cumulative_num));
  }
  return ret;
};

Tensors try_checkpoint(Tensors& inputs) {
  STATS.track("try_checkpoint");
  Tensors ret;
  ret.reserve(inputs.size());
  for (auto& input : inputs) {
    if(input.is_checkpoint()){
      ret.push_back(input);
    }else{
      auto device_id = static_cast<int>(c10::cuda::current_device());
      auto cpt = at::native::checkpoint(input);
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(cpt.unsafeGetTensorImpl());
      auto *pm = getDTBPoolManager();
      pm->lock_temp_ext(device_id, weak(cpti->unsafeGetTensorCell()));
#ifdef DEBUG_MODE
      if(record_op_recs) {
        DTRLogAddress("inner checkpoint "+cpti->unsafeGetTensorCell()->counter_name()+ " " + std::string(cpti->unsafeGetTensorCell()->dtype().name()) + " device:" + std::to_string(device_id), 
          cpti->unsafeGetTensorCell()->pool->addr, cpti->unsafeGetTensorCell()->pool->lock_count);
      }
#endif
      ret.push_back(cpt);
    }
    // ret.push_back(at::native::try_checkpoint(input));
  }
  return ret;
}

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

}
}