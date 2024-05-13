// File: graph_binary.cpp
// -- graph handling source
//-----------------------------------------------------------------------------
// Community detection 
// Based on the article "Fast unfolding of community hierarchies in large networks"
// Copyright (C) 2008 V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre
//
// This program must not be distributed without agreement of the above mentionned authors.
//-----------------------------------------------------------------------------
// Author   : E. Lefebvre, adapted by J.-L. Guillaume
// Email    : jean-loup.guillaume@lip6.fr
// Location : Paris, France
// Time	    : February 2008
//-----------------------------------------------------------------------------
// see readme.txt for more details

#include <sys/mman.h>
#include <fstream>
#include <ATen/checkpoint/DynamicGraph.h>
#include "math.h"

using std::ifstream, std::fstream, std::cout, std::endl;

DynamicGraph::DynamicGraph() {
  nb_nodes     = 0;
  nb_edges     = 0;
  total_weight = 0;
}


DynamicGraph::DynamicGraph(int n, int m, double t, int *d, int *l, float *w) {
/*  nb_nodes     = n;
  nb_links     = m;
  total_weight = t;
  degrees      = d;
  links        = l;
  weights      = w;*/
}


void DynamicGraph::init_from_edges(char *filename, int start_batch, int end_batch, int type){
  // 打开文件
  std::ifstream file(filename);
  
  // 检查文件是否成功打开
  if (!file) {
      std::cerr << "无法打开文件" << std::endl;
      return;
  }
  
  std::string line;
  int cur_line_num = 0;
  nb_edges = 0;
  total_weight = 0;
  // 逐行读取
  while (getline(file, line)) {
      cur_line_num++;
      if(cur_line_num < start_batch) continue;
      if(end_batch != -1 && cur_line_num >= end_batch) break;
      std::istringstream iss(line);
      int a, b, c;
      
      if(type==UNWEIGHTED){
        if (iss >> a >> b) {
            // 确保邻接表足够大
            int max_v = std::max(a, b);
            if (max_v >= edges.size()) {
                edges.resize(max_v + 1);
            }
            
            edges[a].push_back(b);
            edges[b].push_back(a);  // 默认无向图
            nb_edges += 2;
            // add_single_directed_edge(a, b);
            // add_single_directed_edge(b, a);
        }
      }else{
        if (iss >> a >> b >> c) {
            // 确保邻接表足够大
            int max_v = std::max(a, b);
            if (max_v >= edges.size()) {
                edges.resize(max_v + 1);
                weights.resize(max_v + 1);
            }
            
            edges[a].push_back(b);
            edges[b].push_back(a);  // 默认无向图
            weights[a].push_back(c);
            weights[b].push_back(c);
            nb_edges += 2;
        }
      }
  }
  
  // 关闭文件
  file.close();

  nb_nodes = edges.size();
  n2c.resize(nb_nodes);
  for(nid_t i=0; i<nb_nodes; i++) n2c[i] = i;

  // Compute total weight
  for (nid_t i=0 ; i<nb_nodes ; i++) {
    total_weight += (double)weighted_degree(i);
  }
}

void DynamicGraph::display() {
  for (nid_t node=0 ; node<nb_nodes ; node++) {
    auto p = neighbors(node);
    cout << node << ":" ;
    for (nid_t i=0 ; i<nb_neighbors(node) ; i++) {
      if (true) {
	if (!weights.empty())
	  cout << " (" << p.first[i] << " " << p.second[i] << ")";
	else
	  cout << " " << p.first[i];
      }
    }
    cout << endl;
  }
}

void DynamicGraph::display_reverse() {
  for (nid_t node=0 ; node<nb_nodes ; node++) {
    auto p = neighbors(node);
    for (nid_t i=0 ; i<nb_neighbors(node) ; i++) {
      if (node>p.first[i]) {
	if (!weights.empty())
	  cout << p.first[i] << " " << node << " " << p.second[i] << endl;
	else
	  cout << p.first[i] << " " << node << endl;
      }
    }   
  }
}


bool DynamicGraph::check_symmetry() {
  int error=0;
  for (nid_t node=0 ; node<nb_nodes ; node++) {
    auto p = neighbors(node);
    for (nid_t i=0 ; i<nb_neighbors(node) ; i++) {
      nid_t neigh = p.first[i];
      float weight = p.second[i];
      
      auto p_neigh = neighbors(neigh);
      for (nid_t j=0 ; j<nb_neighbors(neigh) ; j++) {
        nid_t neigh_neigh = p_neigh.first[j];
        float neigh_weight = p_neigh.second[j];

        if (node==neigh_neigh && weight!=neigh_weight) {
        cout << node << " " << neigh << " " << weight << " " << neigh_weight << endl;
        if (error++==10)
            exit(0);
        }
      }
    }
  }
  return (error==0);
}


void DynamicGraph::insert_edge(nid_t s, nid_t e, float w) {
  add_single_directed_edge(s, e, w);
  add_single_directed_edge(e, s, w);
}

void DynamicGraph::insert_node_with_edges(nid_t new_node, const vector<nid_t>& ends, const vector<float>& new_weights) {
  assert(new_node>=nb_nodes);
  add_single_node(new_node);
  for(int i=0; i<ends.size(); i++){
    if(new_weights.empty()){
      add_single_directed_edge(new_node, ends[i]);
      add_single_directed_edge(ends[i], new_node);
    }else{
      add_single_directed_edge(new_node, ends[i], new_weights[i]);
      add_single_directed_edge(ends[i], new_node, new_weights[i]);
    }
  }
}

void DynamicGraph::add_from_edges(char *filename, int start_batch, int end_batch, int type){
  // 打开文件
  std::ifstream file(filename);
  // 检查文件是否成功打开
  if (!file) {
      std::cerr << "无法打开文件" << std::endl;
      return;
  }
  
  /// TODO: 接入到runtime的新增节点中
  std::string line;
  int cur_line_num = 0;
  // 逐行读取
  while (getline(file, line)) {
    cur_line_num++;
    if(cur_line_num < start_batch) continue;
    if(end_batch != -1 && cur_line_num >= end_batch) return;
    std::istringstream iss(line);
    int a, b, c;
    // 新的节点会被初始化为独立社区，边直接插入
    if(type==UNWEIGHTED){
      if (iss >> a >> b) {
          add_single_directed_edge(a, b);
          add_single_directed_edge(b, a);
      }
    }else{
      if (iss >> a >> b >> c) {
          add_single_directed_edge(a, b, c);
          add_single_directed_edge(b, a, c);
      }
    }
  }
  
  // 关闭文件
  file.close();

}