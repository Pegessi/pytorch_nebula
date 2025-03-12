#pragma once

#include <vector>
#include <string>
#include <limits>
#include <queue>
#include <unordered_map>
#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/CheckpointTensorCell.h>

namespace c10 {
namespace dtb {

struct DAGNode;
using SDAGNode = intrusive_ptr<DAGNode>;
// using WeakDAGNode = weak_intrusive_ptr<DAGNode>;
using dag_nid_t = long;

struct DAGNode : intrusive_ptr_target {
private:
    bool is_lock = false;
    
public:
    dag_nid_t nid;
    int distance;
    std::vector<std::pair<SDAGNode, int>> out_nodes;
    std::vector<SDAGNode> in_nodes;
    // 这几个部分与weak所记录的可能有些重叠，为了集中处理逻辑，在这里再记录一次
    int in_degree;
    int out_degree;
    int level;
    weak cptc;

    DAGNode(dag_nid_t nid, const weak& cptc);
    std::string to_string() const;
    void lock_node();
    void unlock_node();
    void release_resources() override;
};


struct DynamicDAGShortestPath : intrusive_ptr_target {
private:
    SDAGNode start_node;
    dag_nid_t start_nid;
    std::queue<SDAGNode> queue;
    std::unordered_map<int, SDAGNode> distance_to_max_level_node;
    std::vector<SDAGNode> previous_sorted_snapshot;
    std::vector<SDAGNode> previous_stable_prefix;
    int stable_window_end = 0;


    void _insert_sorted(const SDAGNode& node);
    void _update_sorted_nodes(const SDAGNode& node);

public:
    std::unordered_map<dag_nid_t, SDAGNode> nodes;
    std::vector<SDAGNode> sorted_nodes;

    DynamicDAGShortestPath(dag_nid_t nid, const weak& cptc);
    void add_node(dag_nid_t nid, const weak& cptc);
    void add_edge(dag_nid_t s_id, dag_nid_t t_id, const weak& s, const weak& t, int weight);
    int get_shortest_distance(dag_nid_t nid);
    void relax(const SDAGNode& u, const SDAGNode& v, int weight);
    void process_queue();
    std::vector<SDAGNode> get_sorted_nodes();
    void clear_all_nodes();
    void release_resources() override;
};

using SDAG = intrusive_ptr<DynamicDAGShortestPath>;

struct MultiDAGShortestPaths : intrusive_ptr_target {
private:
    int bid;

public:
    std::vector<SDAG> subgraphs;
    std::unordered_map<dag_nid_t, SDAG> node_to_subgraph;

    explicit MultiDAGShortestPaths(int batch_id);
    void add_node(dag_nid_t nid, const weak& cptc);
    void add_edge(dag_nid_t s_id, dag_nid_t t_id, const weak& s, const weak& t, int weight);
    int get_shortest_distance(dag_nid_t nid);
    std::vector<SDAGNode> get_sorted_nodes_for_subgraph_containing(dag_nid_t nid);
    void clear_all_graphs();
    void release_resources() override;
};

using SMultiDAG = intrusive_ptr<MultiDAGShortestPaths>;


}
}
