import os
import yaml
import traceback

def logger(*args):
    traceback_info = traceback.extract_stack()[-2]
    line_number = traceback_info.lineno
    file_name = traceback_info.filename
    function_name = traceback_info.name
    with open('./gen_log_.txt', 'a') as f:
        res = '[INFO] '+ file_name + ':' + str(line_number) + ' ' + function_name + ' [MSG] '
        msg =  ' '.join(str(arg) for arg in args)
        res += str(msg)
        print(msg)
        f.write(res+'\n')

def get_to_be_impl_ops(yaml_file_path='./to_be_impled.yaml'):
    with open(yaml_file_path, 'r') as f:
        ops_list = yaml.load(f, Loader=yaml.FullLoader)
    logger('Read from {} native functions num:'.format(yaml_file_path), len(ops_list))
    return ops_list

def list_head_files(directory='/home/wangzehua/llm_workspace/dtr_workspace/pytorch_dtr/torch/include/ATen/ops'):
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            res.append({'path': os.path.join(root, file), 'name': file})
            # print(os.path.join(root, file))
    return res

def get_mapped_op_head_files(ops: list, head_files: list):
    """
    @return [...[func_name, head_file_info, register_func_name]...]
    """
    # 获取op名以及调用的函数名
    def get_func_name(opd: dict):
        op_name = str(opd['func']).split('.')[0] if '.' in opd['func'] else str(opd['func']).split('(')[0]
        func_name = str(opd['func']).replace('.', '_').split('(')[0]
        # if 'structured_delegate' in opd.keys():
        #     func_name = opd['structured_delegate'].replace(".", '_')
        if 'autogen' in opd.keys():
            func_name = opd['autogen'].replace(".", '_')
        # if 'dispatch' in opd.keys():
        #     if 'CPU, CUDA, MPS, Meta' in opd['dispatch'].keys():
        #         func_name = opd['dispatch']['CPU, CUDA, MPS, Meta']
        #     if 'CPU, CUDA' in opd['dispatch'].keys():
        #         func_name = opd['dispatch']['CPU, CUDA']
        #     if 'CUDA' in opd['dispatch'].keys():
        #         func_name = opd['dispatch']['CUDA']
        if ',' in func_name:
            func_name = func_name.split(',')[0]
        special_replace_chars = ['Scalar_out', 'Tensor_Scalar_out', 'Tensor_Tensor_out', 'Tensor_Scalar_out', 'Tensor_out']
        for substr in special_replace_chars:
            func_name.replace(substr, 'outf')
        return op_name, func_name
    counts = 0
    not_fully_list = []
    fully_list = []
    
    for op_dict in ops:
        # logger(op_dict)
        op_name, func_name = get_func_name(op_dict)
        # 查询匹配的head_file
        candidates_files = []
        fully_match = False
        for row in head_files: # 先找xxx.h
            head_file_name = row['name']
            target_file_name = op_name[:-1] + '.h' if op_name.endswith('_') else op_name + '.h'
            if target_file_name == head_file_name:
                # logger(op_name, func_name, row)
                candidates_files.append(row)
                fully_match = True
                break
        
        if fully_match:
            # logger(candidates_files[-1])
            # logger(op_name, func_name, candidates_files[-1], op_dict)
            fully_list.append([op_name, func_name, candidates_files[-1], op_dict['func']])
        else:
            # corner case to be deal with
            logger('Not fully match!', op_dict)
            not_fully_list.append(op_dict)
        counts += 1
        # if counts > 6:
        #     break
    logger('Corner case:', len(not_fully_list))
    return fully_list
    

if __name__ == '__main__':
    # head_file_path = '/home/wangzehua/llm_workspace/dtr_workspace/pytorch_dtr/torch/include/ATen/ops'
    # res = list_head_files(head_file_path)
    # print(res[0])
    import re

    # 原始的C++函数声明字符串
    cpp_declaration = '(const at::Tensor & self, int64_t dim=0, c10::optional<int64_t> start=c10::nullopt, c10::optional<int64_t> end=c10::nullopt, int64_t step=1);'

    while '=' in cpp_declaration:
        eq_char_pos = cpp_declaration.index('=')
        sep_char_pos = -1
        for i in range(eq_char_pos, len(cpp_declaration)):
            if cpp_declaration[i] == ',' or cpp_declaration[i] == ')':
                sep_char_pos = i
                break
        print(eq_char_pos, sep_char_pos)
        cpp_declaration = cpp_declaration[:eq_char_pos] + cpp_declaration[sep_char_pos:]
        # break

    # 输出处理后的字符串
    print(cpp_declaration)