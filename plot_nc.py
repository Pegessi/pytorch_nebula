import matplotlib.pyplot as plt
# import networkx as nx
import igraph as ig
import numpy as np
import os
import sys
import warnings
# from matplotlib import cm
from matplotlib import colormaps as cm
import matplotlib.colors as mcolors
import logging


def rgba_to_hex(r, g, b, a):
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    a = int(a * 255)
    hex_r = hex(r)[2:].zfill(2).upper()
    hex_g = hex(g)[2:].zfill(2).upper()
    hex_b = hex(b)[2:].zfill(2).upper()
    hex_a = hex(a)[2:].zfill(2).upper()
    return "#{}{}{}{}".format(hex_r, hex_g, hex_b, hex_a)

def float_to_rgb_coolwarm(x, vmin, vmax):
    x = x+1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')
    rgba = cmap(norm(x))
    return rgba_to_hex(*rgba)
    
def float_to_rgb_lognorm_coolwarm(x, vmin, vmax):
    x = x+1
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')
    rgba = cmap(norm(x))
    return rgba_to_hex(*rgba)

def find_feature_log(log_dir):
    if not os.path.isdir(log_dir):
        print(f"路径 {log_dir} 不是一个有效的文件夹")
        return None
    # 遍历目录下的文件，找到第一个包含 'feature' 的文件
    for f in os.listdir(log_dir):
        full_path = os.path.join(log_dir, f)
        if os.path.isfile(full_path) and 'feature' in f:
            return f  # 找到第一个就返回
    return None  # 如果没找到，返回 None

def ensure_path_and_clear(path_):
    """
    如果路径path_不存在，就创建路径；
    如果路径存在，就清空路径中的所有文件。
    """
    if not os.path.exists(path_):
        # 如果路径不存在，创建路径
        os.makedirs(path_)
        print(f"路径 {path_} 已创建。")
    else:
        # 如果路径存在，清空路径中的所有文件
        for filename in os.listdir(path_):
            file_path = os.path.join(path_, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
            except Exception as e:
                print(f"删除 {file_path} 时出错: {e}")
        print(f"路径 {path_} 中的所有文件已删除。")

def split_log_file(input_log_path, output_folder, marker="xxxxx", M=1):
    """
    拆分log文件为多个新文件，按照规则逐步累积log块进行保存。
    
    参数:
        input_log_path: str - 输入的log文件路径
        output_folder: str - 输出文件夹路径
        marker: str - 标识行的标识符 (默认是"xxxxx")
        M: int - 整除因子 M，必须能整除标识行个数 N
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Step 1: 读取文件并定位标识行及对应块
    with open(input_log_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    marker_indices = [i for i, line in enumerate(lines) if marker in line]
    N = len(marker_indices)
    
    if N % M != 0:
        raise ValueError(f"标识行个数 {N} 无法被 {M} 整除，请调整 M！")
    
    # Step 2: 将N个标识行分隔成N+1块 (最后一块舍弃)
    blocks = []
    start = 0
    for idx in marker_indices:
        blocks.append(lines[start:idx])  # 截取从上一个标识行到当前标识行的内容
        start = idx + 1
    # 舍弃最后一块
    blocks = blocks[:-1]
    
    # Step 3: 按段进行组合拆分
    blocks_per_segment = N // M  # 每段的块数
    total_segments = M
    
    log_count = 1  # 文件编号
    for segment_idx in range(total_segments):
        start_block = segment_idx * blocks_per_segment
        for i in range(1, blocks_per_segment + 1):
            cumulative_blocks = blocks[start_block:start_block + i]
            output_file_path = os.path.join(output_folder, f"{log_count}.log")
            with open(output_file_path, "w", encoding="utf-8") as out_file:
                for block in cumulative_blocks:
                    out_file.writelines(block)  # 写入块的内容
            log_count += 1

    print(f"拆分完成！生成了 {log_count-1} 个log文件，保存到 {output_folder}")
    return N

def get_single_path():
    while True:
        try:
            args = sys.argv[1:]  # 获取命令行参数，忽略脚本名
            if len(args) == 1:
                print(f"接收到路径：{args[0]}")
                return args[0]
            else:
                print("错误：请传入且仅传入一个路径参数。")
                print("使用方式：python script.py <路径>")
                break
        except Exception as e:
            print(f"发生错误: {e}")
            continue

def get_largest_component_size(g, n):
    subgraph = g.subgraph(range(n)) # 提取前 n 个顶点
    components = subgraph.connected_components() # 找到连通子图
    max_component_size = max(len(c) for c in components) # 获取最大连通子图大大小
    return max_component_size


def plot_compute_graph(OPLOG, N2CLOG, FIGURE_DIR, CHECK_DEGREE=6, color_mode=None, plot_comm=False, frontN=None):
    r"""
    读取计算过程中的op日志，解析出对应的计算图并绘制
    """
    with open(OPLOG, 'r') as f:
        data = f.readlines()
        data = [eval(row.replace('\n', '')) for row in data]
    nodes = set()
    edges = []
    nodes_times = {}
    SHOW_CUSTOM_CLUSTER = False
    # CHECK_DEGREE = 6 # gpt 6
    SCALE_UP = 2000
    data = data[:] #  1500 [467:2100]
    # 解析日志得到无向图
    for row in data:
        if row['INSTRUCTION'] != 'INSTRUCTION':
            continue
        if row['name']=='add_' and row['inputs'][0] == row['outputs'][0]:
            continue
        for iid in row['inputs']:
            input_id = int(iid[1:])
            nodes.add(input_id)
            if input_id not in nodes_times.keys():
                nodes_times[input_id] = 0
            for oid in row['outputs']:
                output_id = int(oid[1:])
                nodes_times[input_id] += 1
                edges.append((input_id, output_id, {'op': row['name'], 'cost': row['compute_cost'], 'mem': row['mem_cost']}))
        for oid in row['outputs']:
            output_id = int(oid[1:])
            if output_id not in nodes_times.keys():
                nodes_times[output_id] = 0
            for iid in row['inputs']:
                nodes_times[output_id] += 1
            nodes.add(output_id)
    # print(nodes_times)
    counts = 0
    print(f"Nodes id with degree=={CHECK_DEGREE}")
    for k,v in nodes_times.items():
        if v == CHECK_DEGREE:
            counts+=1
            print(k, end=', ')
    if counts == 0:
        warnings.warn(f"Warning: The number of node with degree {CHECK_DEGREE} is 0!!!", UserWarning)
    else:   
        print('\ntotal', counts)

    # 构造图对象
    g = ig.Graph()
    g.add_vertices(max(list(nodes))+1)
    for ed in edges:
        g.add_edge(ed[0], ed[1])

    ##### custom clusters #####
    if SHOW_CUSTOM_CLUSTER:
        with open(N2CLOG, 'r') as f:
            data = f.readlines()
            data = [int(row.replace('\n','')) for row in data]
        n2c = data[:g.vcount()+1]
        import colorsys
        def distinct_colors(k):
            colors = []
            for i in range(k):
                # 色调分布在0到1之间
                hue = i / k
                # 选择饱和度和亮度为1，以获取鲜艳的颜色
                saturation = 1.0
                value = 1.0
                # 转换为RGB格式
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                # 将RGB从0-1范围转换为0-255范围，并取整
                rgb = tuple(int(x * 255) for x in rgb)
                colors.append(rgb)
            return colors
        g.vs["color"] = distinct_colors(max(n2c))

    # 删除孤立节点 [WARNING] - 这一步后要再重新生成布局才能去正常绘图
    isolated_vertices = [v.index for v in g.vs if g.degree(v.index) == 0]
    g.delete_vertices(isolated_vertices)
    
     
    # 度中心性
    degrees = g.degree()
    max_degree = max(degrees)
    degree_static = { i:0 for i in range(0, max_degree+1) }
    for i in range(len(degrees)):
        degree_static[degrees[i]] += 1
    print(degree_static)
    deg_colors = [float_to_rgb_coolwarm(b, min(degrees), max(degrees)) for b in degrees]
    # deg_colors = ["red" if degree == CHECK_DEGREE else "blue" for degree in degrees]
    print('max degree:', max(degrees))

    # 接近中心性
    closeness = g.closeness()
    closeness = [ 0 if np.isnan(val) else val  for val in closeness]
    clo_colors = [float_to_rgb_lognorm_coolwarm(b, 1+min(closeness), 1+max(closeness)) for b in closeness]
    print('closeness(min, max, mean, median): ', np.min(closeness), np.max(closeness), np.mean(closeness), np.median(closeness))

    # 求解割点
    cut_points = g.articulation_points()
    cut_points = [int(v) for v in cut_points]
    cut_colors = ['red' if i in cut_points else 'blue' for i in range(1, g.vcount()+1)]

    # 介数中心性
    betweenness = g.betweenness()
    bet_dict = { vertex:b for vertex, b in enumerate(betweenness)}
    # logging.info(str(bet_dict))

    sortedVertexIndexList = sorted(range(len(betweenness)), key=lambda v: bet_dict[v], reverse=True) # 对顶点按照介数中心性从大到小排序
    index_map = {v: i for i, v in enumerate(sortedVertexIndexList)} # 生成顶点映射表
    sortedVertexMappingList = [index_map[v] for v in range(len(betweenness))] # 生成排序后的顶点映射列表
    g = g.permute_vertices(sortedVertexMappingList) # 重新排列顶点

    betweenness_sorted = g.betweenness() # 重新计算中心性（验证）
    print('betweeness(min, max, mean, median, max - submax): ', np.min(betweenness), np.max(betweenness), np.mean(betweenness), np.median(betweenness), betweenness_sorted[0] - betweenness_sorted[1])

    size_count_dic = {}
    size_to_first_i = {}  # 新增一个字典，用于得到可能为链的时候对应的子图的最小顶点数i
    for i in range(1, len(betweenness_sorted) + 1):
        size = get_largest_component_size(g, i)
        size_count_dic[size] = size_count_dic.get(size, 0) + 1 # 统计
        # 如果 size 是第一次出现，记录对应的 i
        if size not in size_to_first_i:
            size_to_first_i[size] = i
        log_message = f"由前 {i} 个顶点组成的最大连通子图的大小: {size}"
        logging.info(log_message)  # 将信息记录到日志
    size_count_list = sorted(size_count_dic.items(), key=lambda x: x[1], reverse=True)
    logging.info(f"统计结果: {size_count_list}")  # 将信息记录到日志
    plotPriority = [size_to_first_i[i[0]] for i in size_count_list] # 越靠前，越可能恰好是所求链
    for index, frontN in enumerate(plotPriority[:3]): # 先画三个
        bet_colors = ["red" if i < index else "blue" for i in range(len(betweenness_sorted))]
        g.vs["color"] = bet_colors
        community = g.community_leiden(objective_function='modularity', n_iterations=100)   
        layout = g.layout_kamada_kawai() # 力学布局
        scale_factor = int(g.vcount()/SCALE_UP)
        scale_factor = 1 if scale_factor == 0 else scale_factor # 防止scale_factor为0
        ig.plot(g, os.path.join(FIGURE_DIR, f'graph_betweenness_{index}_front_{frontN}.png'), layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)


    # if frontN: # 给了前N个点
    #     bet_colors = ["red" if i <= frontN else "blue" for i in range(len(betweenness_sorted))] # frontN
    # else:
    #     bet_colors = [float_to_rgb_lognorm_coolwarm(b, 1 + min(betweenness_sorted), 1 + max(betweenness_sorted)) for b in betweenness_sorted]

    return

    # community = g.community_infomap()
    community = g.community_leiden(objective_function='modularity', n_iterations=100)   
    # 使用边介数的层次聚类方法
    # community = g.community_edge_betweenness()
    # 得到实际的社区划分
    # clusters = community.as_clustering()

    layout = g.layout_kamada_kawai()          # 力学布局
    # layout = g.layout_reingold_tilford()        # tree layout
    scale_factor = int(g.vcount()/SCALE_UP)
    scale_factor = 1 if scale_factor == 0 else scale_factor # 防止scale_factor为0
    if SHOW_CUSTOM_CLUSTER:
        ig.plot(g, os.path.join(FIGURE_DIR, 'graph_manual_comm.png'), layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
    
    vcolor_dict = {
        'degree': deg_colors,
        'betweeness': bet_colors,
        'closeness': clo_colors,
        'cut': cut_colors
    }
    if color_mode:
        g.vs["color"] = vcolor_dict[color_mode]
        ig.plot(g, os.path.join(FIGURE_DIR, f'graph_{color_mode}.png'), layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
    else:
        for k,v in vcolor_dict.items():
            g.vs["color"] = v
            ig.plot(g, os.path.join(FIGURE_DIR, f'graph_{k}.png'), layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
    
    if plot_comm:
        ig.plot(community, os.path.join(FIGURE_DIR, './graph_with_comm.png'), layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor, mark_groups=True)

        layout = community.cluster_graph().layout_kamada_kawai()
        ig.plot(community.cluster_graph(), os.path.join(FIGURE_DIR, './comm_graph.png'), layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor, mark_groups=True)
    
    # colors = ["red" if degree == CHECK_DEGREE else "blue" for degree in degrees]
    
    # clusters = ig.VertexClustering(g)
    # clusters._membership = n2c
        
def plot_cumulative_remat_counts(filenames, labels, attr_name, title, visual_percent):
    ay_list = []
    max_range = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            data = f.readlines()
            data = [eval(row.replace('\n', '')) for row in data]
        # {"INSTRUCTION":"INSTRUCTION","cumulative_remat_counts":"1","name":"aten::add.Tensor","rid":"-4661336787295776820"}
        # {"INSTRUCTION":"CONSTANT","NAME":"dependecy counts","VALUE":0}
        ay = []
        for row in data:
            ay.append(int(row[attr_name]))
        # ay_list.append(sorted(ay)[:int(visual_percent * len(ay))])
        ay_list.append(ay)
        print(np.mean(ay_list[-1]), np.min(ay_list[-1]), np.max(ay_list[-1]), np.median(ay_list[-1]))
        max_range = max(max_range, len(ay_list[-1]))
    ax = [i for i in range(1, max_range+1)]
    for i in range(len(ay_list)):
        ay = ay_list[i]
        while len(ay) < max_range:
            ay.append(0)
        plt.bar(ax, ay, label=labels[i])  # 绘制柱状图，调整柱子宽度为0.8
    for li in ay_list:
        for _ in li:
            print(_, end=' ')
        print()
    plt.title(title)  # 添加标题
    plt.xlabel('events index')  # 添加X轴标签
    plt.ylabel('counts')  # 添加Y轴标签

    plt.xticks(rotation=90)  # 旋转X轴标签以提高可读性
    plt.grid(axis='y')  # 添加Y轴网格线

    plt.legend(loc='upper left')
    plt.tight_layout()  # 自动调整子图参数，以充分利用图表空间
    plt.savefig(os.path.join(FIGURE_DIR, 'cunum.jpg'), dpi=600)

def plot_training_loss(files):
    datas = []
    for file_name in files:
        f_data = []
        with open(file_name, 'r') as f:
            data = f.readlines()
            data = [row.replace('\n', '') for row in data]
            for row in data:
                parts = row.split('|')
                consumed_samples = None
                lm_loss = None
                for part in parts:
                    if "consumed samples:" in part:
                        consumed_samples = part.split(':')[1].strip()
                    elif "lm loss:" in part:
                        lm_loss = part.split(':')[1].strip()
                print("Consumed samples:", consumed_samples)
                print("LM loss:", lm_loss)
                if consumed_samples and lm_loss:
                    f_data.append((consumed_samples, lm_loss))
        datas.append(f_data[:1300])
    
    x = [512*2048*i for i in range(1, len(datas[0])+1)]  # batch size=512, seqlen=2048
    y1 = [eval(ele[1]) for ele in datas[0]]
    y2 = [eval(ele[1]) for ele in datas[1]]

    plt.plot(x, y1, label='Megatron-LM')
    plt.plot(x, y2, label='Nebula-Chain')

    plt.title('training convergence')  # 添加标题
    plt.xlabel('tokens')  # 添加X轴标签
    plt.ylabel('training loss')  # 添加Y轴标签

    # plt.xticks(rotation=90)  # 旋转X轴标签以提高可读性
    # plt.grid(axis='y')  # 添加Y轴网格线

    plt.legend(loc='upper right')
    plt.tight_layout()  # 自动调整子图参数，以充分利用图表空间
    plt.savefig(SAVE_PREFIX+'loss.jpg', dpi=600)


    # return datas


if __name__ == '__main__':
    LOGDIR = get_single_path()
    # LOGDIR = '/home/wanghuibing/Accelerate_LLM/Llama_sft/2024-12-16-20-34-11-1428010'

    GRAD_ACCU_STEPS = 2  # 梯度累积步数
    CHECK_DEGREE = 6  # 检查的节点度数
    CALL_FLUSH_TIMES = 0  # 调用flush_community_singleton()次数
    # frontN = 27
    frontN = None

    FIGURE_DIR = os.path.join(LOGDIR, 'figures')  # 图片保存目录
    OPLOG_DIR = os.path.join(LOGDIR, 'opslog')    # OPLOG文件目录
    N2CLOG_DIR = os.path.join(LOGDIR, 'n2clog')   # N2CLOG文件目录
    BETLOG_DIR = os.path.join(LOGDIR, 'betweennessLog')   # N2CLOG文件目录

    # 配置日志
    ensure_path_and_clear(BETLOG_DIR)
    logging.basicConfig(
        filename=os.path.join(BETLOG_DIR, 'betweenness_log_file.log'),  # 设置日志文件名
        level=logging.INFO,       # 设置日志级别
        format='%(asctime)s - %(message)s'  # 设置日志格式
    )

    oplog = find_feature_log(LOGDIR)
    if oplog is None:
        print("未找到OPLOG文件！")
        exit(1)
    oplog = os.path.join(LOGDIR, oplog)

    ensure_path_and_clear(OPLOG_DIR)
    CALL_FLUSH_TIMES = split_log_file(oplog, OPLOG_DIR, "Call flush_community_singleton() times:", GRAD_ACCU_STEPS)

    # for i in range(1, CALL_FLUSH_TIMES + 1):
    for i in [16, 32]:
        print(f"\n\nProcessing {i}...")
        figure_path = os.path.join(FIGURE_DIR, str(i))
        oplog_file = os.path.join(OPLOG_DIR, f"{i}.log")
        n2clog_file = os.path.join(N2CLOG_DIR, f"{i}.txt")
        ensure_path_and_clear(figure_path)
        
        plot_compute_graph(oplog_file, n2clog_file, figure_path, CHECK_DEGREE, 'betweeness', False, frontN)    
    ### 画计算图的
    # plot_compute_graph('./logs/resnet50.log') # resnet50_once.log pp4_ml_gpt.log llama_op_once.log


    # fig_range = [3, 4, 5]
    # fns = [ './logs/remat/remat_counts_' + str(i) + '0%.log' for i in fig_range]
    # labels = [ str(i) + r"0% budget" for i in fig_range]
    # attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'
    # title = 'Cumulatvie Remat Counts' # 'Cumulatvie Remat Counts'
    # visual_percent = 1

    # plot_cumulative_remat_counts(fns, labels, attr_name, title, visual_percent)


    ### 实验说明递归改善的数据
    # fns = [ './logs/remat/remat_counts_30%.log', './logs/remat/remat_nc_counts_30%.log' ]
    # labels = [ r"30% budget dtr", r"30% budget NC" ]
    # attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'
    # title = 'Cumulatvie Remat Counts Comparsion' # 'Cumulatvie Remat Counts'
    # visual_percent = 1

    # plot_cumulative_remat_counts(fns, labels, attr_name, title, visual_percent)

    # plot_training_loss(['./logs/GPT-1.7B_train_log_ml.log', './logs/GPT-1.7B_train_log_nc.log'])