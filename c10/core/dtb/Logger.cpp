#include <c10/core/dtb/Logger.h>
#include <c10/core/dtb/utils.h> // for using during_backward


namespace c10 {
namespace dtb {

std::string time_prefix_ = "";
bool logDirectoryPrepared = false;

void prepareDir_(bool& isDirPrepared, std::string dirName) {
  if (!isDirPrepared) {
    if (std::filesystem::exists(dirName)) {
      for (const auto& entry : std::filesystem::directory_iterator(dirName)) {
        std::filesystem::remove(entry.path());
      }
    } else {
      std::filesystem::create_directory(dirName);
    }
    isDirPrepared = true;
  }
}

// DTRLogger::DTRLogger(std::string suffix_) : time_prefix(get_time_prefix()), out(get_filename(suffix_)) {}
DTRLogger::DTRLogger(const std::string& suffix_) {
  if (time_prefix_.empty()) {
    time_prefix_ = get_time_prefix();
    prepareDir_(logDirectoryPrepared, time_prefix_);
  }

  out = std::ofstream(time_prefix_ + "/" + get_filename(suffix_));
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open log file: " + time_prefix_ + "/" + get_filename(suffix_));
  }
}

DTRLogger2::DTRLogger2() : DTRLogger("feature") {} // 在派生类 DTRLogger2 中调用基类的构造函数并传入 "feature"

// 静态单例方法（派生类版本）
DTRLogger2& DTRLogger2::logger() {
    static DTRLogger2 ret; // 返回DTRLogger2的单例
    return ret;
}

std::string DTRLogger::get_time_prefix() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    pid_t pid = getpid();
    return std::to_string(1900+tm->tm_year) + "-" +
           std::to_string(1+tm->tm_mon) + "-" +
           std::to_string(tm->tm_mday) + "-" +
           std::to_string(tm->tm_hour) + "-" +
           std::to_string(tm->tm_min) + "-" +
          //  std::to_string(tm->tm_sec) + "-" +
           std::to_string(pid);
}

std::string DTRLogger::get_filename(const std::string& name) {
    return time_prefix_ + "-" + name + ".log";
}

void DTRLogger::log(const std::string& str) {
    out << str << std::endl;
}

DTRLogger& DTRLogger::logger() {
    static DTRLogger ret;
    return ret;
}

bool log_json = true;
std::string INSTRUCTION = "INSTRUCTION";
std::string ANNOTATION = "ANNOTATION";
std::string RELEASE = "RELEASE";
std::string PIN = "PIN";
std::string TIME = "TIME";
std::string ARGS = "ARGS";
std::string MEMORY = "MEMORY";
std::string ALIAS = "ALIAS";
std::string NAME = "NAME";
std::string CONSTANT = "CONSTANT";
std::string VALUE = "VALUE";
std::string EVICT = "EVICT";
std::string REMAT = "REMAT";
std::string REMATCOST = "REMAT COST";
std::string ADDR = "ADDR";
std::string DEGREE = "DEGREE";
std::string TENSOR = "TENSOR REC";

void DTRLogCounts(const std::string& name, size_t counts){
  if (log_json){
    json j;
    j[INSTRUCTION] = CONSTANT;
    j[NAME] = name;
    j[VALUE] = counts;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogDepAndCost(const std::string& name, size_t counts, double cost){
  if (log_json){
    json j;
    j[INSTRUCTION] = CONSTANT;
    j[NAME] = name;
    j[VALUE] = counts;
    j[REMATCOST] = cost;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogEvictEvents(const std::string& name, size_t counts){
  if (log_json){
    json j;
    j[INSTRUCTION] = EVICT;
    j[NAME] = name;
    j[VALUE] = counts;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogMemAlloc(size_t alloc, size_t reserved){
  if (log_json){
    json j;
    j[INSTRUCTION] = "COMPARE";
    j["ALLOC"] = alloc;
    j["RESERVE"] = reserved;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " ");
  }
}

void DTRLogEvictAPSEvents(size_t counts){
  if (log_json){
    json j;
    j[INSTRUCTION] = EVICT;
    j[VALUE] = counts;
    DTRLogger::logger().log(j.dump());
  } else {
    // DTRLogger::logger().log(CONSTANT + " " + EVICT);
  }
}

void DTRLogDestructEvents(){
  if (log_json){
    json j;
    j[INSTRUCTION] = "Desturct function called";
    DTRLogger::logger().log(j.dump());
  } else {
    // DTRLogger::logger().log(CONSTANT + " " + EVICT);
  }
}

void DTRLogRematEvents(const std::string& name, size_t counts){
  if (log_json){
    json j;
    j[INSTRUCTION] = REMAT;
    j[NAME] = name;
    j[VALUE] = counts;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogAddress(const std::string& name, uintptr_t addr, size_t memory){
  if (log_json){
    json j;
    j[INSTRUCTION] = TENSOR;
    j[NAME] = name;
    j[ADDR] = addr;
    j[MEMORY] = memory;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogTensorInfo(const std::string& name, uintptr_t addr, size_t memory, size_t degree, double cost, int device){
  if (log_json){
    json j;
    j[INSTRUCTION] = TENSOR;
    j[NAME] = name;
    j[ADDR] = addr;
    j[MEMORY] = memory;
    j[DEGREE] = degree;
    j[REMATCOST] = cost;
    j["device"] = device;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogOPRecords(const int64_t& rid, const std::string& name, const int64_t& compute_cost, size_t &mem_cost, std::vector<std::string> &inputs, std::vector<std::string> &outputs, int device){
   if (log_json){
    json j;
    j[INSTRUCTION] = INSTRUCTION;
    j["rid"] = std::to_string(rid);
    j["name"] = name;
    j["compute_cost"] = std::to_string(compute_cost);
    j["mem_cost"] = std::to_string(mem_cost);
    j["inputs"] = inputs;
    j["outputs"] = outputs;
    j["device"] = device;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogOPRecords(const int64_t& rid, const std::string& name, const int64_t& compute_cost, size_t &mem_cost, std::vector<std::string> &inputs, std::vector<std::string> &outputs, int device, bool if_weight){
   if (log_json){
    json j;
    j[INSTRUCTION] = INSTRUCTION;
    j["rid"] = std::to_string(rid);
    j["name"] = name;
    j["compute_cost"] = std::to_string(compute_cost);
    j["mem_cost"] = std::to_string(mem_cost);
    j["inputs"] = inputs;
    j["outputs"] = outputs;
    j["device"] = device;
    DTRLogger::logger().log(j.dump());
    if (!during_backward && !if_weight)
      DTRLogger2::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
    if (!during_backward && !if_weight)
      DTRLogger2::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogCalculativeRematsRecords(const int64_t& rid, const std::string& name, const int& remat_counts){
   if (log_json){
    json j;
    j[INSTRUCTION] = INSTRUCTION;
    j["rid"] = std::to_string(rid);
    j["name"] = name;
    j["cumulative_remat_counts"] = std::to_string(remat_counts);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogLifeCycle(const std::string& tag, const size_t& org, const size_t& lck, const size_t& remat){
  if (log_json){
    json j;
    j[INSTRUCTION] = "life cycle";
    j["tag"] = tag;
    j["external_count"] = std::to_string(org);
    j["lock_count"] = std::to_string(lck);
    j["remat_count"] = std::to_string(remat);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT);
  }
}

void DTRLogConstant(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = CONSTANT;
    j[NAME] = name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

void DTRLogMemory(const std::string& name, size_t memory) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = MEMORY;
    j[NAME] = name;
    j[MEMORY] = std::to_string(memory);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(name + " " + MEMORY + ": " + std::to_string(memory));
  }
}

void DTRLogApCost(const std::string& name, double cost) {
  if (log_json) {
    json j;
    j[NAME] = name;
    j["cost"] = std::to_string(cost * 1e7);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(name + ": " + std::to_string(cost));
  }
}

void DTRLogAlias(const std::string& name, int index) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = ALIAS;
    j[NAME] = name;
    j[ALIAS] = std::to_string(index);
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(name + " " + ALIAS + ": " + std::to_string(index));
  }
}

void DTRLogAlias(const std::string& name, int index, bool if_weight) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = ALIAS;
    j[NAME] = name;
    j[ALIAS] = std::to_string(index);
    DTRLogger::logger().log(j.dump());
    if (!during_backward)
      DTRLogger2::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(name + " " + ALIAS + ": " + std::to_string(index));
    if (!during_backward)
      DTRLogger2::logger().log(name + " " + ALIAS + ": " + std::to_string(index));
  }
}

void DTRLogCopyFrom(const std::string& to, const std::string& from) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "COPY_FROM";
    j["DST"] = to;
    j["SRC"] = from;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(to + " <- " + from);
  }
}

void DTRLogCopy(const std::string& new_name, const std::string& old_name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "COPY";
    j["DST"] = new_name;
    j["SRC"] = old_name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(new_name + " = " + old_name);
  }
}

void DTRLogMutate(const std::string& name,
                  const std::vector<std::string>& args,
                  const std::vector<size_t>& mutate,
                  const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "MUTATE";
    j[NAME] = name;
    j[ARGS] = args;
    j["MUTATE"] = mutate;
    j[TIME] = time;
    DTRLogger::logger().log(j.dump());
  } else {
    std::string log = name;
    log += "(";
    for (const auto& s : args) {
      log += s;
      log += ", ";
    }
    log += ") ";
    log += " MUTATING: ";
    log += "(";
    for (const size_t i : mutate) {
      log += std::to_string(i);
      log += ", ";
    }
    log += ") ";
    log += TIME;
    log += ": ";
    log += time;
    DTRLogger::logger().log(log);
  }
}

void DTRLogRelease(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = RELEASE;
    j[NAME] = name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(RELEASE + ": " + name);
  }
}

void DTRLogPin(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = PIN;
    j[NAME] = name;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(RELEASE + ": " + name);
  }
}

void DTRLogCall(const std::vector<std::string>& res,
                const std::string& name,
                const std::vector<std::string>& args,
                const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "CALL";
    j[NAME] = name;
    j["RESULT"] = res;
    j[ARGS] = args;
    j[TIME] = time;
    DTRLogger::logger().log(j.dump());
  } else {
    std::string arg = name + "(";
    for (const auto& s : args) {
      arg += s;
      arg += ", ";
    }
    arg += ")";
    std::string log = "(";
    for (const auto& s: res) {
      log += s;
      log += ", ";
    }
    log += ") = ";
    log += arg;
    log += " TIME: ";
    log += time;
    DTRLogger::logger().log(log);
  }
}

void DTRLogMemEvents(const std::string& name, size_t size, int64_t addr){
  if (log_json){
    json j;
    j["TYPE"] = name;
    j["SIZE"] = size;
    j["ADDR"] = addr;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log(CONSTANT + " " + name);
  }
}

}
}