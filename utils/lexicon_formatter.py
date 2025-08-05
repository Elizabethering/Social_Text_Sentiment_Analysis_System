# utils/lexicon_formatter.py

import argparse
import os


def format_lexicon(input_path, output_path):
    """
    读取一个经过人工筛选的、由domain_lexicon_builder.py生成的候选词文件，
    并将其格式化为Jieba可直接使用的、干净的用户词典。

    该函数会提取每行的第一个词（候选词），并忽略注释行和空行。

    :param input_path: (str) 输入的候选词文件路径。
    :param output_path: (str) 输出的、格式化后的用户词典路径。
    """
    print(f"--- 开始格式化候选词文件: {input_path} ---")

    try:
        # 使用 with 语句确保文件能被正确关闭
        with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            processed_count = 0
            for line in f_in:
                # 忽略以'#'开头的注释行和不包含任何内容的空行
                if line.startswith('#') or not line.strip():
                    continue

                # 按空格分割行，并取第一个元素（即候选词）
                # .split() 默认按所有空白字符分割，能处理多个空格的情况
                parts = line.split()
                if parts:
                    word = parts[0]
                    f_out.write(f"{word}\n")
                    processed_count += 1

        print(f"处理完成！已将 {processed_count} 个有效词语写入到最终词典: {output_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_path}")
        print("请确保你已经运行了domain_lexicon_builder.py并生成了候选词文件。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="将由domain_lexicon_builder.py生成的、经过人工筛选的候选词文件，格式化为Jieba可用的标准用户词典。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="输入的候选词文件路径。\n"
             "例如: data/weibo_train_data_discovered_words_raw.txt"
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="输出的、格式化后的最终用户词典路径。\n"
             "例如: data/weibo_final_user_dict.txt"
    )

    args = parser.parse_args()

    # 调用主函数
    format_lexicon(input_path=args.input, output_path=args.output)
