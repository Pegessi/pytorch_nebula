#include <c10/core/dtb/MemGraph.h>

namespace c10 {
namespace dtb {

MemBlockTwin::MemBlockTwin(int device, size_t size) : device(device), size(size) {
    MemSegmentTwin* new_seg = new MemSegmentTwin(this);
    segment = new_seg;
}

MemBlockTwin::MemBlockTwin(int device, size_t size, MemSegmentTwin* segment, MemBlockTwin* pre, MemBlockTwin* next) 
    : device(device), size(size), segment(segment), pre(pre), next(next) {
    
}


MemSegmentTwin::MemSegmentTwin(MemBlockTwin* block) : device(block->device), total_size(block->size), head_addr(block->addr) {
    blocks.insert(block);
    /// TODO: add aps
}

void MemSegmentTwin::insert(MemBlockTwin* block) {
    assert(block!=nullptr);
    blocks.insert(block);
}

void MemSegmentTwin::erase(MemBlockTwin* block) {
    assert(block!=nullptr);
    blocks.erase(block);
}

bool MemSegmentTwin::release() {
    for (auto block : blocks) {
        // 首先检查每个 block 的前一个和后一个 block，解除它们的关联
        if (block->pre) {
            block->pre->next = block->next;
        }
        if (block->next) {
            block->next->pre = block->pre;
        }

        // 如果当前 block 也在 segment 的成员变量中，需要特别处理
        if (block->segment == this) {
            block->segment = nullptr;
        }

        // 安全释放内存
        delete block;
    }
    // 清空 blocks 集合
    blocks.clear();

    return true; // 返回 true 表示释放操作成功执行
}

// MemBlockTwin* MemGraph::get_allocated_block(uintptr_t ptr) {
//     auto it = allocated_blocks.find(ptr);
//     if(it==allocated_blocks.end()) return nullptr;
//     return it->second;
// }


}
}