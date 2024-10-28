#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/core/dtb/utils.h>
#include <c10/cuda/dtb/DTBManager.h>
#include <queue>

#define TORCH_CHECK(a, ...)   // replace original TORCH_CHECK  profile mode

namespace c10 {
namespace dtb {


#ifdef ORIGINAL_DTR
void CheckpointTensorCell::fill(const Tensor& t) {
  STATS.track("CheckpointTensorCell::fill");
  if (!(this->t)) {
    pool->set_addr(get_addr(t));
    this->t = std::make_unique<Tensor>(std::move(t));
    pool->set_not_evicted(pool);                          /// TAG: 改变标志位，更新cost, MEM_FIRST_EVICT在上面的函数中更新了p2ap(add_ap)
    if (!defined) {                                       /// 这里是将所有的属性拷贝一遍，满足兼容性
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
#else
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
    pool->set_addr(get_addr(t));
    this->t = std::make_unique<Tensor>(std::move(t));
    pool->set_not_evicted(pool);                          /// TAG: 改变标志位，更新cost, MEM_FIRST_EVICT在上面的函数中更新了p2ap(add_ap)
    if (!defined) {                                       /// 这里是将所有的属性拷贝一遍，满足兼容性
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
#endif

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
#ifdef MEM_FIRST_EVICT
  auto *pm = getDTBPoolManager();
  pm->remove_p2ap(pool->addr);            // remove old ptr record
#endif
#ifdef DEBUG_MODE
  if(trace_register_and_release){
    printf("---|cptc evict, addr:%ld\n", reinterpret_cast<uintptr_t>(t->data_ptr()));
  }
  if(record_op_recs) {
    if(t)
      DTRLogAddress("release "+counter_name() + " if_tmp:"+std::to_string(pool->if_temp?1:0) + " if_weight:" + std::to_string(pool->if_weight?1:0)
         + " if_retain:" + std::to_string(pool->is_retain?1:0)
         + " " + std::to_string(pool->external_count) + std::to_string(pool->lock_count), 
        reinterpret_cast<uintptr_t>(t->data_ptr()), pool->memory);
    else
      DTRLogAddress("release "+counter_name() + " if_tmp:"+std::to_string(pool->if_temp?1:0) + " if_weight:" + std::to_string(pool->if_weight?1:0) 
         + " if_retain:" + std::to_string(pool->is_retain?1:0)
         + std::to_string(pool->external_count) + std::to_string(pool->lock_count), 
        pool->addr, pool->memory);
  }
#endif
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
      // TORCH_CHECK(remat);
#ifdef DEBUG_MODE
      if(!remat){
        printf("[CHECK REMAT ERROR] tid:%s if_temp:%d device:%d tensors:%ld lc:%ld ec:%ld rc:%ld\n", counter_name().c_str(),
          pool->if_temp ? 1 : 0, pool->device_id, pool->tensors.size(),
          pool->lock_count, pool->external_count, pool->remat_count);
        printStackTrace();
      }
#endif
      TORCH_INTERNAL_ASSERT(remat);
#ifdef DEBUG_MODE
      // if(record_er_counts)
      //   DTRLogRematEvents(counter_name(), 0);
      if(record_op_recs)
        DTRLogAddress("remat need "+counter_name(), pool->addr, pool->memory);
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

/**
 * Remat output of cptc'remater
 */
void CheckpointTensorCell::remat_neghibors(int remat_depth) {
  if(remat && pool->external_count>0)
    get();  // remat self
  if(remat_depth > 0) {
    for(auto &wcptc: pool->neighbors) {
      if(auto scptc = wcptc.lock()) {
          // if(remat && pool->external_count>0)
          //   scptc->get();
          scptc->remat_neghibors(remat_depth-1);
      }
    }
  }
}

}
}