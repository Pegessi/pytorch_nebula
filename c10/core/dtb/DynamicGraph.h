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

namespace c10 {
namespace dtb {

#define WEIGHTED   0
#define UNWEIGHTED 1

// #define USE_CSR

using nid_t = unsigned int;
using namespace std;

class DynamicGraph {
 private:
  inline void add_single_node(nid_t new_node){
    if(new_node<nb_nodes) return;
    // cerr << "before add info ---- new node:" << new_node << " nb_nodes:" << nb_nodes << "|" << edges.size() << endl;
    n2c.resize(new_node+1);
    if(!weights.empty()) weights.resize(new_node+1);

    // 原有的nb_nodes逻辑上是number of nodes + 1，因此这里从nb_nodes开始计算是合理的
    for(int i = nb_nodes; i <= new_node; i++){
      /* adj table format */
      edges.push_back({});
      // n2c[i] = i;
      n2c[i] = new_node;  // 将空结点归属到新节点上
      if(!weights.empty()) weights.push_back({});
#ifdef USE_CSR
      /* update csr format */
      degrees.emplace_back(degrees.back()); 
#endif
    }
    nb_nodes = edges.size();    /// [TAG] edges[0]是空载的，如果有node id是0，那么不空载
    // cerr << "after add info ---- new node:" << new_node << " nb_nodes:" << nb_nodes << "|" << edges.size() << endl;
  }

  inline void add_single_directed_edge(int s, int e, int w=1) {
    /* adj table format */
    add_single_node(s);
    add_single_node(e);
    edges[s].emplace_back(e);
    if(!weights.empty()){
      weights[s].emplace_back(w);
    }
#ifdef USE_CSR
    /* update csr format */
    int insert_position = degrees[s + 1];
    links.insert(links.begin() + insert_position, e);
    for (int i = s + 1; i < degrees.size(); i++) {
        degrees[i]++;
    }
#endif

    nb_edges += 1;
    total_weight += w;
  }

 public:
  nid_t nb_nodes;           // number of nodes
  unsigned long nb_edges;   // number of edges
  double total_weight;      // ∑w_i

  vector<unsigned long> degrees;          // accumulative degree, deg(k) = degrees[k]-degrees[k-1] [CSR row_ptr]
  vector<nid_t> links;                    // links id, ex: node 0 has three neighbor 3,4,5 then links[0-2]=3,4,5. len(links) = nb_links = 2∑deg [CSR col_ind]
  vector<vector<nid_t>> edges;            // edges id, adjacency table format of links [Adj TABLE]
  vector<vector<float>> weights;          // weight of link(if WEIGHTED)               [Adj TABLE]
  
  vector<int> n2c;                      // community that node belongs to

  DynamicGraph();

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
  void init_from_edges(char *filename, int start_batch, int end_batch, int type);

  void add_from_edges(char *filename, int start_batch, int end_batch, int type);
  
  DynamicGraph(int nb_nodes, int nb_links, double total_weight, int *degrees, int *links, float *weights);

  void display(void);
  void display_reverse(void);
  void display_binary(char *outfile);
  bool check_symmetry();
  
  // insert a new edge between existed two nodes
  void insert_edge(nid_t start, nid_t end, float weight = 1.);

  // insert a new node and relative edges of new node
  void insert_node_with_edges(nid_t new_node, const vector<nid_t>& ends, const vector<float>& weights);

  // return the number of neighbors (degree) of the node
  inline nid_t nb_neighbors(nid_t node);

  // return the number of self loops of the node
  inline double nb_selfloops(nid_t node);

  // return the weighted degree of the node
  inline double weighted_degree(nid_t node);

  // return pointers to the first neighbor and first weight of the node
  inline pair<vector<nid_t>&, vector<float>&> neighbors(nid_t node);
};

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
        nb_loop += weights.empty() ? 1. : static_cast<double>(p.second[i]);
    }
  }
  return nb_loop;
}

inline double DynamicGraph::weighted_degree(nid_t node) {
  assert(node>=0 && node<nb_nodes);

  if (weights.empty())
    return (double)nb_neighbors(node);
  else {
    auto p = neighbors(node);
    double res = 0;
    for (nid_t i=0 ; i<nb_neighbors(node) ; i++) {
      res += static_cast<double>(p.second[i]);
    }
    return res;
  }
}

inline pair<vector<nid_t>&, vector<float>&> DynamicGraph::neighbors(nid_t node) {
  assert(node>=0 && node<nb_nodes);
  return {edges[node], weights[node]};
}

}
}

