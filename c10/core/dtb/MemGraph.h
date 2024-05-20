#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {
namespace dtb {

struct MemBlockTwin;
struct MemSegmentTwin;
// using StrongMemBlock = intrusive_ptr<MemBlockTwin>;
// using WeakMemBlock = weak_intrusive_ptr<MemBlockTwin>;

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
    MemBlockTwin(int device, size_t size, MemSegmentTwin* segment, MemBlockTwin* pre, MemBlockTwin* next);
};

/**
 * Record some memory block in a constant physical memory segment.
*/
struct MemSegmentTwin {
    int device;
    size_t total_size;
    uintptr_t head_addr;

    bool can_be_evict;
    // time_t last_used_time;   // time record by ap
    ska::flat_hash_set<MemBlockTwin*> blocks;
    ska::flat_hash_set<WeakAP> own_aps;

    // first creat a block and create a segment about it
    MemSegmentTwin(MemBlockTwin* block);

    void insert(MemBlockTwin* block);
    void erase(MemBlockTwin* block);

    void insert(const StrongAP& ap);
    void erase(const StrongAP& ap);

    bool evictable();

    /**
     * Release all blocks in this segment when this segment is physically released.
     * Then should delete this MemSegmentTwin like `delete mst;`
    */
    bool release();

    void display();
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

    // MemBlockTwin* get_allocated_block(uintptr_t ptr);        // BUG: 重名
    
};



}
}