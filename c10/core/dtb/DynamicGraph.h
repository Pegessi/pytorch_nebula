#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/CheckpointTensorCell.h>

namespace c10 {
namespace dtb {

// #define USE_CSR

using nid_t = unsigned int;
using namespace std;

#define CANNOT_EVICT_CM_VAL 1e-36

struct SingletonDGNode : intrusive_ptr_target {
  nid_t nid;
  weak value;
  float cm_val=CANNOT_EVICT_CM_VAL;   // c/m
  size_t mem=0;
  bool is_lock = false;
  SingletonDGNode(nid_t id, const weak& weak_cell);

  // bool is_equal(const StrongDGNode& other) const {
  //     return value == other->value;
  // }

  void lock_value();

  void unlock_value();

  void release_resources() override;

};

using StrongDGNode = intrusive_ptr<SingletonDGNode>;


struct DynamicGraph : public intrusive_ptr_target {
 private:
  bool weighted=false;

  inline void add_single_node(nid_t new_node, const weak& wcptc){
    // skip inserted node
    if(new_node<nb_nodes) return;
    // cerr << "before add info ---- new node:" << new_node << " nb_nodes:" << nb_nodes << "|" << edges.size() << endl;
    n2c.resize(new_node+1);
    cptcs.resize(new_node+1);
    // if(weighted) weights.resize(new_node+1);
    weights.resize(new_node+1);

    // 原有的nb_nodes逻辑上是number of nodes + 1，因此这里从nb_nodes开始计算是合理的
    for(int i = nb_nodes; i <= new_node; i++){
      /* adj table format */
      edges.push_back({});
      // n2c[i] = i;
      n2c[i] = new_node;  // 将空结点归属到新节点上
      // if(weighted) weights.push_back({});
      weights.push_back({});
      cptcs[i] = StrongDGNode::make(new_node, wcptc);

    }
    nb_nodes = edges.size();    /// [TAG] edges[0]是空载的，如果有node id是0，那么不空载
    // cerr << "after add info ---- new node:" << new_node << " nb_nodes:" << nb_nodes << "|" << edges.size() << endl;
  }

  inline void add_single_directed_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w=1) {
    /* adj table format */
    add_single_node(s, s_cell);
    add_single_node(e, e_cell);
    edges[s].emplace_back(e);
    // if(weighted){
    TORCH_INTERNAL_ASSERT(s < weights.size())
    weights[s].emplace_back(w);
    // }

    nb_edges += 1;
    total_weight += w;
  }

 public:
  nid_t nb_nodes;           // number of nodes
  unsigned long nb_edges;   // number of edges
  double total_weight;      // ∑w_i

  vector<unsigned long> degrees;          // accumulative degree, deg(k) = degrees[k]-degrees[k-1] [CSR row_ptr]
  vector<nid_t> links;                    // links id, ex: node 0 has three neighbor 3,4,5 then links[0-2]=3,4,5. len(links) = nb_links = 2∑deg [deprecated] [CSR col_ind]
  vector<vector<nid_t>> edges;            // edges id, adjacency table format of links [Adj TABLE]
  vector<vector<float>> weights;          // weight of link(if WEIGHTED)               [Adj TABLE]
  vector<StrongDGNode> cptcs;                     // id to cptc, original graph should maintain this.
  
  vector<int> n2c;                        // community that node belongs to

  DynamicGraph();
  DynamicGraph(int type);

  // binary file format is
  // 4 bytes for the number of nodes in the graph
  // 8*(nb_nodes) bytes for the cumulative degree for each node:
  //    deg(0)=degrees[0]
  //    deg(k)=degrees[k]-degrees[k-1]
  // 4*(sum_degrees) bytes for the links
  // IF WEIGHTED 4*(sum_degrees) bytes for the weights in a separate file
  DynamicGraph(char *filename, char *filename_w, int type);

  /**
   * Read graph from file which has rows owning a pair of vertex expressing a edge
   * if WEIGHTED, file lines like [start end weight] else [start end]
  */
  // void init_from_edges(char *filename, int start_batch, int end_batch, int type);

  // void add_from_edges(char *filename, int start_batch, int end_batch, int type);
  
  DynamicGraph(int nb_nodes, int nb_links, double total_weight, int *degrees, int *links, float *weights);

  void display(void);
  void display_reverse(void);
  void display_binary(char *outfile);
  bool check_symmetry();

  bool is_border_node(nid_t node);
  inline bool is_weighted_graph() { return weighted; }
  // insert a new edge between existed two nodes
  void insert_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w=1.);

  // insert a new node and relative edges of new node
  void insert_node_with_edges(nid_t new_node, const weak& s_cell, const vector<nid_t>& ends, const vector<weak>& e_cells, const vector<float>& new_weights);

  // return the number of neighbors (degree) of the node
  inline nid_t nb_neighbors(nid_t node);

  // return the number of self loops of the node
  inline double nb_selfloops(nid_t node);

  // return the weighted degree of the node
  inline double weighted_degree(nid_t node);

  // return pointers to the first neighbor and first weight of the node
  inline pair<vector<nid_t>&, vector<float>&> neighbors(nid_t node);

  void release_resources() override;
};

using StrongDG = intrusive_ptr<DynamicGraph>;

inline nid_t DynamicGraph::nb_neighbors(nid_t node) {
  assert(node>=0 && node<nb_nodes);
  return static_cast<nid_t>(edges[node].size());
}

inline double DynamicGraph::nb_selfloops(nid_t node) {
  assert(node>=0 && node<nb_nodes);
  auto p = neighbors(node);
  double nb_loop = 0.;
  for (nid_t i=0 ; i<nb_neighbors(node) ; i++) {
    if(p.first[i]==node){
        nb_loop += static_cast<double>(p.second[i]);
    }
  }
  return nb_loop;
}

inline double DynamicGraph::weighted_degree(nid_t node) {
  assert(node>=0 && node<nb_nodes);

  // if (!weighted)
  //   return (double)nb_neighbors(node);
  // else {
  auto p = neighbors(node);
  double res = 0;
  for (nid_t i=0 ; i<nb_neighbors(node) ; i++) {
    res += static_cast<double>(p.second[i]);
  }
  return res;
  // }
}

inline pair<vector<nid_t>&, vector<float>&> DynamicGraph::neighbors(nid_t node) {
  assert(node>=0 && node<nb_nodes);
  return {edges[node], weights[node]};
}

}
}

