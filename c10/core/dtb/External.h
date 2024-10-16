#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/core/dtb/Rematerializer.h>


namespace c10{
namespace dtb{


// An external reference.
// Each strong will have at most one external reference.
// By keeping such an invariant, whenever an external reference die,
// We know that the underlying strong is only used internally.
// Thus, when it die we can apply optimization like banishing/infinite staleness.
// We keep this invariant by only allowing CheckpointTensorImpl to make new External,
// When new CheckpointTensorImpl is constructed.
struct External : intrusive_ptr_target {
  External(const strong& value);

  External(Tensor& value, bool if_weight=false);

  External(Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat);

  strong value;
  
  void release_resources() override;

};

}
}