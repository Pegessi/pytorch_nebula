
#include <sys/mman.h>
#include <fstream>
#include "DynamicGraph.h"
#include "math.h"

namespace c10 {
namespace dtb {

SingletonDGNode::SingletonDGNode(nid_t id, const weak& weak_cell) : nid(id), value(weak_cell) {}

void SingletonDGNode::lock_value(){
  if(!is_lock){
    if(auto cell = value.lock()){
      store_in_special_pool[cell->pool->device_id] = true;
      if(cell->defined)  // remove cell firstly
      {
        auto t_ = cell->t->clone(); 
        cell->pool->evict(0);
        cell->fill(t_);
      }else{
        cell->get();
      }
      store_in_special_pool[cell->pool->device_id] = false;
      cell->pool->is_retain = true;
      cell->pool->lock();
      is_lock = true;
    }
  }
}

void SingletonDGNode::unlock_value() {
  if(is_lock){
    if(auto cell = value.lock()){
      cell->pool->is_retain = false;
      is_lock = false;
      cell->pool->unlock();
    }
  }
}

void SingletonDGNode::release_resources() {
  unlock_value();
  value.reset();
}

DynamicGraph::DynamicGraph() {
  nb_nodes     = 0;
  nb_edges     = 0;
  total_weight = 0;
}

DynamicGraph::DynamicGraph(char *filename, char *filename_w, int type) {
  ifstream finput;
  finput.open(filename,fstream::in | fstream::binary);

  // Read number of nodes on 4 bytes
  finput.read((char *)&nb_nodes, 4);
  assert(finput.rdstate() == ios::goodbit);

  // Read cumulative degree sequence: 8 bytes for each node
  // cum_degree[0]=degree(0); cum_degree[1]=degree(0)+degree(1), etc.
  degrees.resize(nb_nodes);
  finput.read((char *)&degrees[0], nb_nodes*8);

  // Read links: 4 bytes for each link (each link is counted twice)
  nb_edges=degrees[nb_nodes-1];
  edges.resize(nb_nodes);
  n2c.resize(nb_nodes);
  for(nid_t i=0; i<nb_nodes; i++) n2c[i] = i;
  links.resize(nb_edges);
  finput.read((char *)(&links[0]), (long)nb_edges*8);  // 读入id即为node id，对于不连续的部分作为孤立结点处理

  // csr格式的links转换为邻接表存储
  for(int i=0; i<degrees[0]; i++){
    edges[0].emplace_back(links[i]);
  }
  for(int nid=1; nid<nb_nodes; nid++){
    int deg = degrees[nid] - degrees[nid-1];
    for(int i=0; i<deg; i++){
        edges[nid].emplace_back(links[degrees[nid-1]+i]);
    }
  }

  // IF WEIGHTED : read weights: 4 bytes for each link (each link is counted twice)
  weights.resize(0);
  total_weight=0;
  if (type==WEIGHTED) {
    ifstream finput_w;
    finput_w.open(filename_w,fstream::in | fstream::binary);
    weights.resize(nb_edges);
    finput_w.read((char *)&weights[0], (long)nb_edges*4);  
  }    

  // Compute total weight
  for (nid_t i=0 ; i<nb_nodes ; i++) {
    total_weight += (double)weighted_degree(i);
  }
}

DynamicGraph::DynamicGraph(int n, int m, double t, int *d, int *l, float *w) {
/*  nb_nodes     = n;
  nb_links     = m;
  total_weight = t;
  degrees      = d;
  links        = l;
  weights      = w;*/
}

DynamicGraph::DynamicGraph(int type) {
  if(type==WEIGHTED) weighted = true;
  nb_nodes     = 0;
  nb_edges     = 0;
  total_weight = 0; 
}

/*
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
 */

void DynamicGraph::display_binary(char *outfile) {
  ofstream foutput;
  foutput.open(outfile ,fstream::out | fstream::binary);

  foutput.write((char *)(&nb_nodes),4);
  foutput.write((char *)(&degrees[0]),4*nb_nodes);
  foutput.write((char *)(&links[0]),8*nb_edges);
}

void DynamicGraph::insert_edge(nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w) {
  add_single_directed_edge(s, e, s_cell, e_cell, w);
  add_single_directed_edge(e, s, e_cell, s_cell, w);
}

void DynamicGraph::insert_node_with_edges(nid_t new_node, const weak& s_cell, const vector<nid_t>& ends, const vector<weak>& e_cells, const vector<float>& new_weights) {
  assert(new_node>=nb_nodes);
  add_single_node(new_node, s_cell);
  for(int i=0; i<ends.size(); i++){
    if(!weighted){
      add_single_directed_edge(new_node, ends[i], s_cell, e_cells[i]);
      add_single_directed_edge(ends[i], new_node, e_cells[i], s_cell);
    }else{
      add_single_directed_edge(new_node, ends[i], s_cell, e_cells[i], new_weights[i]);
      add_single_directed_edge(ends[i], new_node, e_cells[i], s_cell, new_weights[i]);
    }
  }
}

/*
void DynamicGraph::add_from_edges(char *filename, int start_batch, int end_batch, int type){
  // 打开文件
  std::ifstream file(filename);
  // 检查文件是否成功打开
  if (!file) {
      std::cerr << "无法打开文件" << std::endl;
      return;
  }
  
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
*/


bool DynamicGraph::is_border_node(nid_t node) {
  size_t cur_c = n2c[node];
  auto dg_node = cptcs[node];
  if(auto scptc = dg_node->value.lock()) {
    for(auto neigh_cell: scptc->pool->neighbors) {
      if(auto cell = neigh_cell.lock()) {
        // 这里存在邻居是上一个图的节点的情况？
        if(cell->dg_id >= n2c.size()) {
          // std::cout << "dg id out of n2c size! [" << cell->dg_id << ", " << n2c.size() << "] where nb_nodes=" << nb_nodes << "\n";
          return true;
        }
        // TORCH_INTERNAL_ASSERT(cell->dg_id<n2c.size());
        size_t n_c = n2c[cell->dg_id];
        if(n_c != cur_c) return true;
      }
    }
  }
  return false;
}

void DynamicGraph::release_resources() {
  
}

}
}