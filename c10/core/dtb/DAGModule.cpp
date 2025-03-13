#include <c10/core/dtb/DAGModule.h>

namespace c10 {
namespace dtb {

#pragma region DAGNode
DAGNode::DAGNode(dag_nid_t nid, const weak& cptc) : nid(nid), distance(std::numeric_limits<int>::max()), in_degree(0), out_degree(0), level(0), cptc(cptc) {}

std::string DAGNode::to_string() const {
    return "DAGNode(" + std::to_string(nid) + ")";
}

void DAGNode::lock_node() {
    if(!is_lock){
        if(auto cell = cptc.lock()){
            // store_in_special_pool[cell->pool->device_id] = true;
            // if(cell->defined)  // remove cell firstly
            // {
            //     auto t_ = cell->t->clone(); 
            //     cell->pool->evict(0);
            //     cell->fill(t_, true);
            // }else{
            //     cell->get();
            // }
            // store_in_special_pool[cell->pool->device_id] = false;
            cell->get();
            cell->pool->is_retain = true;
            cell->pool->lock();
            is_lock = true;
        }
    }
}

void DAGNode::unlock_node() {
    if(is_lock){
        if(auto cell = cptc.lock()){
            cell->pool->is_retain = false;
            cell->pool->unlock();
            is_lock = false;
        }
    }
}

void DAGNode::release_resources() {
    in_nodes.clear();
    out_nodes.clear();
    cptc.reset();
}

#pragma endregion



#pragma region DynamicDAGShortestPath
// DynamicDAGShortestPath 实现
DynamicDAGShortestPath::DynamicDAGShortestPath(dag_nid_t nid, const weak& cptc) {
    start_nid = nid;
    start_node = SDAGNode::make(nid, cptc);
    start_node->nid = nid;
    start_node->distance = 0;
    start_node->level = 0;
    nodes[nid] = start_node;
    sorted_nodes.push_back(start_node);
    distance_to_max_level_node[0] = start_node;
    distance_to_last_change_time[0] = get_current_time();
}

void DynamicDAGShortestPath::add_node(dag_nid_t nid, const weak& cptc) {
    if (nodes.find(nid) == nodes.end()) {
        SDAGNode new_node = SDAGNode::make(nid, cptc);
        new_node->nid = nid;
        nodes[nid] = new_node;
    }
}

void DynamicDAGShortestPath::_insert_sorted(const SDAGNode& node) {
    auto it = sorted_nodes.begin();
    while (it != sorted_nodes.end()) {
        SDAGNode current = *it;
        if (node->distance < current->distance) {
            break;
        } else if (node->distance == current->distance) {
            return;
        }
        ++it;
    }
    sorted_nodes.insert(it, node);
    distance_to_max_level_node[node->distance] = node;
    distance_to_last_change_time[node->distance] = get_current_time();
    operation_counter++;
    if (operation_counter%DAG_UPDATE_STABLE_STRIDE==0 && operation_counter > DAG_GRAPH_CONSTRAINT_SIZE) _update_stable_window();
}

void DynamicDAGShortestPath::_update_stable_window(bool final) {
    if(final) {
        // 锁定前95%的node
        int lock_count = 0;
        int total_count = sorted_nodes.size(), lock_max_idx = total_count * 0.95;
        for (int i = 0; i < total_count; ++i) {
            if (i < lock_max_idx) {
                sorted_nodes[i]->lock_node();
                total_lock_counts++;
            } else {
                sorted_nodes[i]->unlock_node();
            }
        }
        return;
    }
    // 获取按照distance_to_last_change_time的访问时间排序的keys，访问时间越早位置越靠前
    std::vector<int> keys;
    keys.reserve(distance_to_last_change_time.size());
    for (const auto& pair : distance_to_last_change_time) {
        keys.push_back(pair.first);
    }
    std::sort(keys.begin(), keys.end(), [&](int a, int b) {
        return distance_to_last_change_time[a] < distance_to_last_change_time[b];
    });
    // 获取这些对应distance的node
    std::vector<SDAGNode> time_order_nodes;
    time_order_nodes.reserve(keys.size());
    for (auto& key : keys) {
        time_order_nodes.push_back(distance_to_max_level_node[key]);
    }
    if(!last_timer_order_nodes.empty()){
        // 寻找第一个不同的node的index
        int idx = 0;
        for (; idx < time_order_nodes.size(); ++idx) {
            if (last_timer_order_nodes[idx] != time_order_nodes[idx]) {
                break;
            }
        }
        if (idx > last_same_idx) {
            for(int i=last_same_idx; i<idx; ++i){
                time_order_nodes[i]->lock_node();
                total_lock_counts++;
            }
        }else if (idx < last_same_idx) {
            for(int i=idx; i<last_same_idx; ++i){
                time_order_nodes[i]->unlock_node();
                total_unlock_counts++;
            }
        }
        last_same_idx = idx;
    }
    last_timer_order_nodes = time_order_nodes;
}

void DynamicDAGShortestPath::_update_sorted_nodes(const SDAGNode& node) {
    if (distance_to_max_level_node.find(node->distance) != distance_to_max_level_node.end()) {
        SDAGNode old_node = distance_to_max_level_node[node->distance];
        if (old_node != node) {
            auto it = std::find(sorted_nodes.begin(), sorted_nodes.end(), old_node);
            old_node->unlock_node();
            total_unlock_counts++;
            if (it != sorted_nodes.end()) {
                sorted_nodes.erase(it);
            }
        }
    }
    _insert_sorted(node);
}


void DynamicDAGShortestPath::add_edge(dag_nid_t s_id, dag_nid_t t_id, const weak& s, const weak& t, int weight) {
    add_node(s_id, s);
    add_node(t_id, t);
    SDAGNode u = nodes[s_id];
    SDAGNode v = nodes[t_id];
    u->out_nodes.emplace_back(v, weight);
    v->in_nodes.emplace_back(u);
    u->out_degree++;
    v->in_degree++;

    v->level = -1;
    for (const auto& predecessor : v->in_nodes) {
        v->level = std::max(v->level, predecessor->level);
    }
    v->level++;
    if (u->distance != std::numeric_limits<int>::max()) {
        int prev_dist = v->distance;
        relax(u, v, weight);
        v->level = -1;
        for (const auto& predecessor : v->in_nodes) {
            v->level = std::max(v->level, predecessor->level);
        }
        v->level++;
        if (v->distance != prev_dist || v->level != v->level) {
            _update_sorted_nodes(v);
        }
    }
}

void DynamicDAGShortestPath::relax(const SDAGNode& u, const SDAGNode& v, int weight) {
    if (v->distance > u->distance + weight) {
        int prev_dist = v->distance;
        v->distance = u->distance + weight;
        if (prev_dist != v->distance) {
            queue.push(v);
        }
    }
}

void DynamicDAGShortestPath::process_queue() {
    while (!queue.empty()) {
        SDAGNode u = queue.front();
        queue.pop();
        for (const auto& [v, weight] : u->out_nodes) {
            int prev_dist = v->distance;
            int prev_level = v->level;
            v->level = -1;
            for (const auto& predecessor : v->in_nodes) {
                v->level = std::max(v->level, predecessor->level);
            }
            v->level++;
            relax(u, v, weight);
            v->level = -1;
            for (const auto& predecessor : v->in_nodes) {
                v->level = std::max(v->level, predecessor->level);
            }
            v->level++;
            if (v->distance != prev_dist || v->level != prev_level) {
                _update_sorted_nodes(v);
            }
        }
    }
}

int DynamicDAGShortestPath::get_shortest_distance(dag_nid_t nid) {
    auto it = nodes.find(nid);
    return it != nodes.end() ? it->second->distance : std::numeric_limits<int>::max();
}

std::vector<SDAGNode> DynamicDAGShortestPath::get_sorted_nodes() {
    return sorted_nodes;
}

void DynamicDAGShortestPath::clear_all_nodes() {
    for(auto& node: sorted_nodes) {
        node->unlock_node();
    }
}

void DynamicDAGShortestPath::release_resources() {
    start_node.reset();
    nodes.clear();
    sorted_nodes.clear();
    distance_to_max_level_node.clear();
    distance_to_last_change_time.clear();
}

#pragma endregion

#pragma region MultiDAGShortestPaths
MultiDAGShortestPaths::MultiDAGShortestPaths(int bid) { bid = bid; }

void MultiDAGShortestPaths::add_node(dag_nid_t nid, const weak& cptc) {
    if (node_to_subgraph.find(nid) == node_to_subgraph.end()) {
        SDAG new_subgraph = SDAG::make(nid, cptc);
        subgraphs.push_back(new_subgraph);
        node_to_subgraph[nid] = new_subgraph;
    }
}

void MultiDAGShortestPaths::add_edge(dag_nid_t s_id, dag_nid_t t_id, const weak& s, const weak& t, int weight) {
    add_node(s_id, s);
    add_node(t_id, t);
    SDAG u_subgraph = node_to_subgraph[s_id];
    SDAG v_subgraph = node_to_subgraph[t_id];

    if (u_subgraph != v_subgraph) { // merge graph
        if (u_subgraph->get_sorted_nodes().size() < v_subgraph->get_sorted_nodes().size()) {
            std::swap(u_subgraph, v_subgraph);
        }
        for (const auto& [nid, node] : v_subgraph->nodes) {
            u_subgraph->add_node(nid, node->cptc);
            node->unlock_node();    // unlock small graph nodes
            node_to_subgraph[nid] = u_subgraph;
            for (const auto& [neighbor, edge_weight] : node->out_nodes) {
                u_subgraph->add_edge(nid, neighbor->nid, node->cptc, neighbor->cptc, edge_weight);
            }
        }
        auto it = std::find(subgraphs.begin(), subgraphs.end(), v_subgraph);
        if (it != subgraphs.end()) {
            subgraphs.erase(it);
        }
        u_subgraph->process_queue();
    }
    u_subgraph->add_edge(s_id, t_id, s, t, weight);
}

int MultiDAGShortestPaths::get_shortest_distance(dag_nid_t nid) {
    auto it = node_to_subgraph.find(nid);
    return it != node_to_subgraph.end() ? it->second->get_shortest_distance(nid) : std::numeric_limits<int>::max();
}

std::vector<SDAGNode> MultiDAGShortestPaths::get_sorted_nodes_for_subgraph_containing(dag_nid_t nid) {
    auto it = node_to_subgraph.find(nid);
    if (it != node_to_subgraph.end()) {
        return it->second->get_sorted_nodes();
    }
    return std::vector<SDAGNode>();
}

void MultiDAGShortestPaths::clear_all_graphs() {
    for(auto& subgraph: subgraphs) {
        subgraph->clear_all_nodes();
    }
}

void MultiDAGShortestPaths::release_resources() {
    subgraphs.clear();
    node_to_subgraph.clear();
}

#pragma endregion

}
}
