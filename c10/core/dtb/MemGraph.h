#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {
namespace dtb {

// struct MemBlockTwin;
// using StrongMemBlock = intrusive_ptr<MemBlockTwin>;
// using WeakMemBlock = weak_intrusive_ptr<MemBlockTwin>;

// struct MemSegmentTwin;
// using StrongMemSegment = intrusive_ptr<MemSegmentTwin>;
// using WeakMemSegment = weak_intrusive_ptr<MemSegmentTwin>;


using StrongAP = intrusive_ptr<AliasPool>;
using WeakAP = weak_intrusive_ptr<AliasPool>;


/**
 * Twin object for Block in cudaCachingAllocator.
 * 
*/
struct MemBlockTwin {
    int device;
    size_t size;
    uintptr_t addr;
    MemSegmentTwin* segment{nullptr};
    MemBlockTwin* pre{nullptr};
    MemBlockTwin* next{nullptr};
    
    MemBlockTwin(int device, size_t size);
    MemBlockTwin(int device, size_t size, MemSegmentTwin* segment);
};

/**
 * Record some memory block in a constant physical memory segment.
*/
struct MemSegmentTwin {
    int device;
    size_t total_size;
    uintptr_t head_addr;

    bool can_be_evict;
    time_t last_used_time;
    ska::flat_hash_set<MemBlockTwin*> blocks;
    ska::flat_hash_set<WeakAP> own_aps;

    // first creat a block and create a segment about it
    MemSegmentTwin(const MemBlockTwin* block);

    void insert(const MemBlockTwin* block);
    void insert(const StrongAP& ap);

    void erase(const MemBlockTwin* block);
    void erase(const StrongAP& ap);

    bool evictable();
    /**
     * Release all blocks in this segment.
     * And reserve this segment.
    */
    bool evict();
};


struct MemGraph
{
    // Get block by ptr_value
    ska::flat_hash_map<uintptr_t, MemBlockTwin*> allocated_blocks;
    void add_allocated_block(MemBlockTwin* block) {
        allocated_blocks[block->addr] = block;
    }
    void erase_allocated_block(MemBlockTwin* block) {
        auto it = allocated_blocks.find(block->addr);
        allocated_blocks.erase(it);
    }
    
    // Only record blocks being used
    std::unordered_map<uintptr_t, MemBlockTwin*> active_blocks;
};



}
}