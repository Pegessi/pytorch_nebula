#include <c10/core/dtb/ResidualChain.h>
#include <c10/hip/HIPCachingAllocator.h>

#define TORCH_CHECK(a, ...) 

namespace c10 {
namespace dtb {

#pragma region ResidualChain

ChainNode::ChainNode(const weak& weak_cell) : value(weak_cell) {}

void ChainNode::lock_value(){
  if(!is_lock){
    if(auto cell = value.lock()){
      store_in_special_pool[cell->pool->device_id] = true;
      if(cell->defined)  // remove cell firstly
      {
        auto t_ = cell->t->clone(); 
        cell->pool->evict(0);
        cell->fill(t_);
      }else{
        cell->get();
      }
      store_in_special_pool[cell->pool->device_id] = false;
      cell->pool->is_retain = true;
      cell->pool->lock();
      is_lock = true;
    }
  }
}

void ChainNode::unlock_value() {
  if(is_lock){
    if(auto cell = value.lock()){
      cell->pool->is_retain = false;
      is_lock = false;
      cell->pool->unlock();
    }
  }
}

void ChainNode::release_resources() {
  unlock_value();
  value.reset();
}

ResidualChain::ResidualChain(const StrongChainNode& n) : root(n) {
  members.push_back(n);
}

void ResidualChain::insert(const StrongChainNode& n) {
  n->parent = root;
  members.push_back(n);

  if(size()>CHAIN_LENGTH_LOCK_THRESHOLD) {  // 认为该链是要找的链
    // printf("[TAG] with chain len:%ld\n", size());
    is_locked = true;
    for(int i = last_check_idx; i<size(); i++){
      if(i%CHAIN_LOCK_STRIDE==0)
        members[i]->lock_value();
    }
    last_check_idx = size() - 1;
  }
}

void ResidualChain::erase(const StrongChainNode& n) {
  auto it = std::find(members.begin(), members.end(), n);
  if(it!=members.end())
    members.erase(it);
}

bool ResidualChain::in_chain(const StrongChainNode& n){
  const auto& last_node = members.back();
  return last_node->is_equal(n);
}

void ResidualChain::clear_members() {
  for(auto &node: members){
    node->unlock_value();
  }
  members.clear();
}

void ResidualChain::release_resources() {
  members.clear();
  root.reset();
}

#pragma endregion


}
}