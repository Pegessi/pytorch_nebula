#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/CheckpointTensorCell.h>

namespace c10{
namespace dtb{

enum KeyChainStatus {
  IS_RIGHT_CHAIN,   // 是要找的链
  TO_BE_DELETE,     // 需要删除该链
  NORMAL            // 正常状态
};

struct ChainNode;
using StrongChainNode = intrusive_ptr<ChainNode>;
using WeakChainNode = weak_intrusive_ptr<ChainNode>;

struct ChainNode : intrusive_ptr_target {
  weak value;
  bool is_lock = false;
  mutable intrusive_ptr<ChainNode> parent;
  ChainNode(const weak& weak_cell);

  bool is_equal(const StrongChainNode& other) const {
      return value == other->value;
  }

  void lock_value();

  void unlock_value();

  void release_resources() override;

};

// constexpr const int CHAIN_LENGTH_LOCK_THRESHOLD = 8;  // 16
// constexpr const int CHAIN_LOCK_STRIDE = 4;

// TODO: weak并不能作为键
struct ResidualChain : intrusive_ptr_target {
  StrongChainNode root;
  std::vector<StrongChainNode> members;
  int last_check_idx = 0;
  bool is_locked = false;

  ResidualChain(const StrongChainNode& n);

  size_t size(){
    return members.size();
  }

  void insert(const StrongChainNode& n);

  void erase(const StrongChainNode& n);

  bool in_chain(const StrongChainNode& n);

  void clear_members();

  void release_resources() override;

};

using ResidualChainRef = intrusive_ptr<ResidualChain>;


}
}