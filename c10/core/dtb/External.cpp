#include <c10/core/dtb/External.h>
#include <c10/core/dtb/utils.h>

#define TORCH_CHECK(a, ...) 

namespace c10 {
namespace dtb {

#pragma region ExternalMethods

External::External(const strong& value) : value(value) {
  value->pool->register_external();                       /// TAG: Aliaspool引用计数的唯一增加入口
}

External::External(Tensor& value, bool if_weight) :
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
          ) {} /// static_cast<int>(value.device().index()) 存在无device的tensor, probably empty tensor

External::External(Tensor& value,
          const intrusive_ptr<AliasPool>& pool,
          const intrusive_ptr<Rematerializer>& remat) :
  External(strong::make(value, pool, remat)) { }

void External::release_resources() {    /// TAG: Aliaspool引用计数的唯一减少入口
    // printf("%s %d %ld %d ex:%ld\n", value->counter_name().c_str(), ((value->pool->memory > 0 && (!value->pool->ecn) && value->pool->head_remat)||(value->pool->memory > 0&& value->pool->head_remat==nullptr && !value->pool->if_weight)) ? 1 : 0, value->pool->memory, value->pool->if_weight ? 1 : 0, value->pool->external_count);
    // printf("pool of %s release_external finish.\n", value->counter_name().c_str());
#ifdef DEBUG_MODE
  if(trace_register_and_release){
    printf("release ext, which pool device:%d counts-ext:%ld lock:%ld remat:%ld\n", value->pool->device_id,
      value->pool->external_count, value->pool->lock_count, value->pool->remat_count);
  }
#endif
  value->pool->release_external();
  value.reset();
}

#pragma endregion

}
}