#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {
namespace dtb {

class ISegmentTwin {
public:
    virtual ~ISegmentTwin() {}
    // virtual void insert(Block* block) = 0;
    // virtual void erase(Block* block) = 0;
    virtual bool empty() const = 0;
    virtual size_t total_size() const = 0;
    virtual time_t last_change_time() const = 0;
};

class ISegmentManager {
public:
    virtual ~ISegmentManager() {}
    // virtual void add_block2segment(Block* block, ISegmentTwin* seg) = 0;
    // virtual ISegmentTwin* get_segment_of_block(void* ptr, bool remove = false) = 0;
    // virtual void insert(ISegmentTwin* new_segment) = 0;
    // virtual void update(ISegmentTwin* segment) = 0;
    // virtual void erase(ISegmentTwin* segment) = 0;
    virtual void display_segments() = 0;
    /**
     * 需要能根据给的地址/ptr，获取一组候选块，并检索是否可驱逐
     * 若不满足，需要继续获取新的一组块，直到可满足？
    */
};



}
}