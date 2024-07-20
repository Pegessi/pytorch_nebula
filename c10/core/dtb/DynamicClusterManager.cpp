#include <c10/core/dtb/DynamicClusterManager.h>

namespace c10 {
namespace dtb {

DCManager::DCManager(int init_size, int inteval, int nbp, double minm, int type) : 
                    cluster_init_size(init_size), cluster_interval(inteval), nb_pass(nbp), min_modularity(minm) {
    original_dg = StrongDG::make(type);
}

void DCManager::insert_single_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w=1) {
    auto orig_nb_nodes = original_dg->nb_nodes;
    original_dg->insert_edge(s, e, s_cell, e_cell, w);
    if(com.defined())
        com->insert_edge(s, e, s_cell, e_cell, original_dg, w);     // here dynamic set community of new node

    if(original_dg->nb_nodes > orig_nb_nodes) {
        grow_size++;
    }

    if(C10_UNLIKELY(!com.defined())) {
        if(grow_size >= cluster_init_size) {
            com = StrongCOM::make(*(original_dg.get()), nb_pass, min_modularity);
            // run_Louvain_Detection(...);      // 这里的参数需要考虑如何传
            grow_size = 0;
        }
    } else {
        if(grow_size >= cluster_interval) {
            // run_Louvain_Detection(...);      // 这里的参数需要考虑如何传
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
        if(0) {
            cerr << "Continue Level:" << ++accum_run_level << endl;
            cerr << "  Before Louvain, network size: " 
                << com->g.nb_nodes << " nodes, " 
                << com->g.nb_edges << " links, "
                << com->g.total_weight << " weight." << endl;
        }
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

        if (0)
        {
            cerr << "  modularity increased from " << cur_modularity << " to " << new_mod << endl;
            cerr << "  network size: " 
                << c.g.nb_nodes << " nodes, " 
                << c.g.nb_edges << " links, "
                << c.g.total_weight << " weight." << endl;
        }

        cur_modularity = new_mod;
        if (0)
            cerr << "  end computation\n";

    } while(improvement);
}



}
}