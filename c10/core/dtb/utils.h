#pragma once

#include <c10/core/dtb/comm_heads.h>

namespace c10 {
namespace dtb {

struct PerfStats;

struct Timer {
  std::string name;
  Time start;
  Timer(std::string name, Time start);
  Timer() {}
  ~Timer();
};

constexpr bool stats = true;

struct PerfStats {
  using TimerStats = std::tuple<std::string, Time, Time, Duration>;
  Time start;
  std::unordered_map<std::string, int> calls;
  std::vector<PerfStats::TimerStats> timers;

  PerfStats();

  void track(const char*) { }

  ~PerfStats();
};

extern PerfStats STATS;


size_t memory(const Tensor& t);
size_t get_addr(const Tensor& t);

void set_global_memory_budget(size_t budget);

void registerStreamLabel(c10::Stream stream, int label);

int getStreamLabel(hipStream_t stream);

Tensor uncheckpoint(const strong& input);

Tensors uncheckpoint(const strongs& inputs);

Tensors uncheckpoint_with_depth(const strongs& inputs, int& cumulative_num);

Tensors try_checkpoint(Tensors& inputs);

void printStackTrace();

}
}