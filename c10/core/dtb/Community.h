#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <c10/core/dtb/comm_heads.h>

#include "DynamicGraph.h"

namespace c10 {
namespace dtb {

using namespace std;

class SingletonCommunity {
  private:
    std::unordered_map<StrongDGNode, bool> border_marker;
    bool is_lock=false;

  public:
    SingletonCommunity() {}

    std::set<StrongDGNode> inner_nodes;
    std::set<StrongDGNode> border_nodes;

    void insert_node(const StrongDGNode& new_node, bool is_border);
    void erase_node(const StrongDGNode& new_node);
    void lock_borders();
    void unlock_borders();
    void clear_outers(const StrongDG& og, int cid);

};

// using StrongSingleCOM = intrusive_ptr<SingletonCommunity>;


struct Community : intrusive_ptr_target {

  vector<double> neigh_weight;
  vector<unsigned int> neigh_pos;
  unsigned int neigh_last;

  DynamicGraph g;     // network to compute communities for
  int size;           // nummber of nodes in the network and size of all vectors
  vector<int> n2c;    // community to which each node belongs
  vector<double> in;  // sum of the weights of the links inside C, ∑w_c
  vector<double> tot; // sum of the weights of the links incident to nodes in C, ∑deg(N_c)

  // number of pass for one level computation
  // if -1, compute as many pass as needed to increase modularity
  int nb_pass;
  

  // a new pass is computed if the last one has generated an increase 
  // greater than min_modularity
  // if 0. even a minor increase is enough to go for one more pass
  double min_modularity;

  // init community from a graph with graph copy constructor
  Community (DynamicGraph g, int nb_pass, double min_modularity);

  /**
   * Extend community with initial state and extend graph from file
  */
  void add_from_edges(char *filename, int start_batch, int end_batch, int type, DynamicGraph* og);

  void insert_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, const StrongDG& og, float w=1.);

  // display the community of each node
  void display();

  // remove the node from its current community with which it has dnodecomm links
  inline void remove(int node, int comm, double dnodecomm);

  // insert the node in comm with which it shares dnodecomm links
  inline void insert(int node, int comm, double dnodecomm);

  // insert a new node in comm which maybe new too
  inline void insert_new(int node, int comm, double dnodecomm);

  /**
    compute the gain of modularity if node where inserted in comm
    given that node has dnodecomm links to comm.  The formula is:
    >>> [(In(comm)+2d(node,comm))/2m - ((tot(comm)+deg(node))/2m)^2] - [In(comm)/2m - (tot(comm)/2m)^2 - (deg(node)/2m)^2]
    where
    >>> In(comm)    = number of half-links strictly inside comm
    >>> Tot(comm)   = number of half-links inside or outside comm (sum(degrees))
    >>> d(node,com) = number of links from node to comm
    >>> deg(node)   = node degree
    >>> m           = number of links
  */
  inline double modularity_gain(int node, int comm, double dnodecomm, double w_degree);

  // compute the set of neighboring communities of node.
  // for each community, gives the number of links from node to comm(neigh_pos & neigh_weight)
  void neigh_comm(unsigned int node);

  // compute the modularity of the current partition
  double modularity();

  // displays the graph of communities as computed by one_level
  void partition2graph();
  // displays the current partition (with communities renumbered from 0 to k-1)
  void display_partition();

  // generates the binary graph of communities as computed by one_level
  DynamicGraph partition2graph_binary(const StrongDG& og);

  // compute communities of the graph for one level
  // return true if some nodes have been moved
  bool one_level();
};

using StrongCOM = intrusive_ptr<Community>;

inline void Community::remove(int node, int comm, double dnodecomm) {
  assert(node>=0 && node<size);

  tot[comm] -= g.weighted_degree(node);               // comm的总权重减少(内外)
  in[comm]  -= 2*dnodecomm + g.nb_selfloops(node);    // comm内的权重减少
  n2c[node]  = -1;
}

inline void Community::insert(int node, int comm, double dnodecomm) {
  assert(node>=0 && node<size);

  tot[comm] += g.weighted_degree(node);
  in[comm]  += 2*dnodecomm + g.nb_selfloops(node);
  n2c[node]  = comm;
}

inline void Community::insert_new(int node, int comm, double dnodecomm){
  // cerr << node << " " << size << endl;
  assert(node==size); // 新增单例社区
  size += 1;
  n2c.emplace_back(comm);
  tot.emplace_back(2*dnodecomm + g.nb_selfloops(node));
  in.emplace_back(g.weighted_degree(node));
  neigh_weight.resize(size,-1);
  neigh_pos.resize(size);
}

inline double Community::modularity_gain(int node, int comm, double dnodecomm, double w_degree) {
  assert(node>=0 && node<size);

  double totc = (double)tot[comm];
  double degc = (double)w_degree;
  double m2   = (double)g.total_weight;
  double dnc  = (double)dnodecomm;
  
  return (dnc - totc*degc/m2);
}

}
}

