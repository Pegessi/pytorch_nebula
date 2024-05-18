#include <c10/core/dtb/MemGraph.h>

namespace c10 {
namespace dtb {

MemBlockTwin::MemBlockTwin(int device, size_t size) : device(device), size(size) {
    MemSegmentTwin* new_seg = new MemSegmentTwin(this);
    segment = new_seg;
}

MemBlockTwin::MemBlockTwin(int device, size_t size, MemSegmentTwin* segment) : device(device), size(size), segment(segment) {}


MemSegmentTwin::MemSegmentTwin(const MemBlockTwin* block) : device(block->device), total_size(block->size), head_addr(block->addr) {

}

}
}