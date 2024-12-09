#pragma once
#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/core/dtb/DynamicGraph.h>
#include <c10/core/dtb/Community.h>
#include <c10/core/dtb/ResidualChain.h>

namespace c10 {
namespace dtb {

struct DCManager : intrusive_ptr_target {
private:
    StrongDG original_dg;
    StrongCOM com;
    vector<SingletonCommunity> singleton_comms;    // record communities detected, designed for single forward computation graph
    std::vector<ResidualChainRef> chains;
    int cluster_init_size;      // nb_nodes > cluster_init_size, then initial com
    int cluster_interval;       // △nb_nodes > cluster_interval, then dynamic change com
    int nb_pass;                // for com
    double min_modularity;      // for com
    int type;                   // for dg, WEIGHTED or UNWEIGHTED

    size_t grow_size=0;           // The number of nodes that have increased since the last clustering
    double cur_modularity;
    size_t accum_run_level=0;

    size_t accum_lock_mem = 0;
public:
    explicit DCManager(int init_size, int inteval, int nbp, double minm, int type);

    void insert_single_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w=1.);

    void run_Louvain_Detection();

    void flush_community_singleton();

    void clear_comms();

    void release_resources() override;
    
};

using StrongDCM = intrusive_ptr<DCManager>;

}
}