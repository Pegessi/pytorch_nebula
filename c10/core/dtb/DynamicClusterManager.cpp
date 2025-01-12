#include <c10/core/dtb/DynamicClusterManager.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace c10 {
namespace dtb {

DCManager::DCManager(int init_size, int inteval, int nbp, double minm, int type) : 
                    cluster_init_size(init_size), cluster_interval(inteval), nb_pass(nbp), min_modularity(minm) {
    original_dg = StrongDG::make(type);
}

void DCManager::insert_single_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w) {
    auto orig_nb_nodes = original_dg->nb_nodes;
    original_dg->insert_edge(s, e, s_cell, e_cell, w);
    if(com.defined())
        com->insert_edge(s, e, s_cell, e_cell, original_dg, w);     // here dynamic set community of new node

    if(original_dg->nb_nodes > orig_nb_nodes) {
        grow_size++;
    }

    if(C10_LIKELY(com.defined())) {
        if(grow_size >= cluster_interval) {
            run_Louvain_Detection();
            flush_community_singleton();
            grow_size = 0;
        }
    } else {
        if(grow_size >= cluster_init_size) {
            DynamicGraph g = *original_dg;
            com = StrongCOM::make(g, nb_pass, min_modularity);
            run_Louvain_Detection();
            flush_community_singleton();
            grow_size = 0;
        }
    }
}

void DCManager::run_Louvain_Detection(){
    // time_t time_begin, time_end;
    // time(&time_begin);
    bool improvement=true;
    double new_mod;
    // DynamicGraph g = c.g;
    do {
    #ifdef DEBUG_MODE
        if(record_dcr_process) {
            cerr << "Continue Level:" << ++accum_run_level << endl;
            cerr << "  Before Louvain, network size: " 
                << com->g.nb_nodes << " nodes, " 
                << com->g.nb_edges << " links, "
                << com->g.total_weight << " weight." << endl;
        }
    #endif
        improvement = com->one_level();    // 进行一轮合并操作，包括结点
        // if(level==0){
        //   org = c.g;
        // }
        new_mod = com->modularity();
        // if (level==display_level)
        //     c.g.display();
        // if (display_level==-1)
        //     c.display_partition();
        DynamicGraph new_g = com->partition2graph_binary(original_dg);   // 生成社区为新结点的新图，进行下一轮合并
        com = StrongCOM::make(new_g, nb_pass, min_modularity);

    #ifdef DEBUG_MODE
        if (record_dcr_process)
        {
            cerr << "  modularity increased from " << cur_modularity << " to " << new_mod << endl;
            cerr << "  network size: " 
                << com->g.nb_nodes << " nodes, " 
                << com->g.nb_edges << " links, "
                << com->g.total_weight << " weight." << endl;
        }
    #endif

        cur_modularity = new_mod;

    #ifdef DEBUG_MODE
        if (record_dcr_process)
            cerr << "  end computation\n";
    #endif

    } while(improvement);
}


void DCManager::flush_community_singleton() {
    size_t unlock_counts = 0;
    if(singleton_comms.size()!=com->size)
        singleton_comms.resize(com->size);
    for(size_t nid=0; nid<original_dg->nb_nodes; nid++) {
        auto cid = original_dg->n2c[nid];
        bool is_border = original_dg->is_border_node(nid);
        if(!is_border) {    // 非border解锁
            if(original_dg->cptcs[nid]->is_lock) {
                original_dg->cptcs[nid]->unlock_value();
                accum_lock_mem -= original_dg->cptcs[nid]->value.lock()->pool->memory;
                unlock_counts++;
            }
        } 

        singleton_comms[cid].insert_node(original_dg->cptcs[nid], is_border);
    }
    size_t total_mem = 0, total_act = 0;
    for(size_t cid=0; cid<singleton_comms.size(); cid++) {
        singleton_comms[cid].clear_outers(original_dg, cid);
        accum_lock_mem += singleton_comms[cid].lock_borders();
#ifdef DEBUG_MODE
        if(record_dcr_process) {
            if(singleton_comms[cid].inner_nodes.size()>0 || singleton_comms[cid].border_nodes.size()>0) {
                size_t can_inner = 0, can_border = 0;
                size_t border_mem = 0, border_act_mem = 0;
                for(auto& node: singleton_comms[cid].inner_nodes) {
                    if(auto scptc = node->value.lock()) {
                        if(scptc->pool->evictable()) can_inner++;
                    }
                }
                vector<size_t> top_mems;
                for(auto& node: singleton_comms[cid].border_nodes) {
                    if(auto scptc = node->value.lock()) {
                        if(scptc->pool->evictable()) {
                            can_border++;
                            border_act_mem += scptc->pool->memory;
                        }
                        border_mem += scptc->pool->memory;
                        top_mems.emplace_back(scptc->pool->memory);
                    }
                }
                total_mem += border_mem;
                total_act += border_act_mem;
                std::cout << "flush singleton[" << cid << "] inners: " << singleton_comms[cid].inner_nodes.size() << " borders: " << singleton_comms[cid].border_nodes.size() 
                        << ", can evict ratio:" << static_cast<float>(can_inner) / static_cast<float>(singleton_comms[cid].inner_nodes.size()) 
                        << "|" << static_cast<float>(can_border) / static_cast<float>(singleton_comms[cid].border_nodes.size())
                        << " border total memory: " << border_mem/1024/1024 << " MB, and activation border is: " << border_act_mem/1024/1024 << " MB, top-5 border mem: ";
                for(auto& mem: top_mems) std::cout << mem/1024/1024 << " ";
                std::cout << "\n";

            }
        }

        if(record_dcr_memory) {
            std::vector<void*> ptrs;
            for(auto& node: singleton_comms[cid].inner_nodes) {
                if(auto scptc = node->value.lock()) {
                    if(scptc->defined)
                        ptrs.push_back(scptc->t->data_ptr());
                }
            }
            for(auto& node: singleton_comms[cid].border_nodes) {
                if(auto scptc = node->value.lock()) {
                    if(scptc->defined)
                        ptrs.push_back(scptc->t->data_ptr());
                }
            }
            c10::cuda::CUDACachingAllocator::logPtrInfo(c10::cuda::current_device(), ptrs);
        }
#endif
    }
#ifdef DEBUG_MODE
    if(record_dcr_process) {
        std::cout << "[flush_community_singleton summary] " << accum_lock_mem/1024/1024 << " MB, lock counts: " << dcr_lock_counts <<"\n";
        std::cout << "flush attain " << singleton_comms.size() << " singleton community, border total mem: " << total_mem/1024/1024 
                << " MB, and activation border is: " << total_act/1024/1024 << " MB\n";
        dcr_lock_counts = 0;
    }
#endif
}

void DCManager::clear_comms() {
    for(auto& sc: singleton_comms) {
        sc.unlock_borders();
    }
}

void DCManager::release_resources() {
    clear_comms();
    singleton_comms.clear();
    original_dg.reset();
    com.reset();
}

}
}