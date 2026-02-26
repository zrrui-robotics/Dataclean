import argparse
import os
import shutil
from pathlib import Path

def extract_and_rename_jsons(source_dir, output_dir, prefix_number):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 1. 确保输出目录存在，如果不存在则自动创建
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. 查找源目录及其所有子目录下的 .json 文件
    json_files = list(source_path.rglob("*.json"))
    
    if not json_files:
        print("未在指定目录中找到任何 JSON 文件！")
        return

    print(f"共找到 {len(json_files)} 个 JSON 文件，开始提取并重命名...\n")

    # 3. 遍历提取并复制
    for n, file_path in enumerate(json_files, start=1):
        # 这里的 n 是一个递增的数字 (1, 2, 3...)
        # 如果你想让 N 是文件所在的“父文件夹名称”，请注释掉上面那行 for 循环，改为下面这样：
        # for file_path in json_files:
        #     n = file_path.parent.name
        
        # 拼接新的文件名
        new_filename = f"{prefix_number}-{n}.json"
        destination = output_path / new_filename
        
        # 使用 shutil.copy2 复制文件（保留原文件的创建时间和权限等元数据）
        shutil.copy2(file_path, destination)
        print(f"成功提取: {file_path.relative_to(source_path)}  =>  {new_filename}")

    print("\n✅ 所有文件提取完成！")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="从源目录及子目录提取 JSON 并重命名为 前缀-序号.json")
    p.add_argument("--source-dir", "-s", default=None, help="要搜索的源文件夹路径")
    p.add_argument("--output-dir", "-o", default=None, help="输出文件夹路径（默认与 source-dir 相同）")
    p.add_argument("--prefix", "-p", default="57", help="文件名前缀数字，如 57 得到 57-1.json, 57-2.json")
    args = p.parse_args()
    if args.source_dir is None:
        p.error("请指定 --source-dir（或在 benchmark.sh 中设置 SOURCE_DIR）")
    source_dir = args.source_dir
    output_dir = args.output_dir if args.output_dir is not None else source_dir
    extract_and_rename_jsons(source_dir, output_dir, args.prefix)