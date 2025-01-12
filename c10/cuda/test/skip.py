def filter_content(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    keep_lines = []
    skip = False
    for line in lines:
        if line.strip() == "#ifdef GMLAKE_ENABLE":
            skip = True
            continue
        elif line.strip() == "#endif":
            skip = False
            continue
        if not skip:
            keep_lines.append(line)

    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(keep_lines)


if __name__ == "__main__":
    input_file_path = "/data/wangzehua/pytorch_nebula/c10/cuda/CUDACachingAllocator.cpp"  # 替换为实际的输入文件路径
    output_file_path = "/data/wangzehua/pytorch_nebula/c10/cuda/test/test.cpp"  # 替换为实际的输出文件路径
    filter_content(input_file_path, output_file_path)