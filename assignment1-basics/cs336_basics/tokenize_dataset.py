import numpy as np
import time 
import tqdm
from cs336_basics.tokenizer import Tokenizer 
from cs336_basics.pretokenization_example  import find_chunk_boundaries 
import multiprocessing

tokenizer = Tokenizer.from_files("tinystories_vocab.pkl","tinystories_merges.pkl",special_tokens =["<|endoftext|>"])


def process_chunk(file_path:str, start, end):
    with open(file_path, "rb") as f:
        #找到起始位置
        f.seek(start)

        #计算取的长度
        context_length = end -start

        #读取context_length的二进制
        chunk_bytes = f.read(context_length)

        #将二进制字节解压解码为UTF-8字符串
        text = chunk_bytes.decode("utf-8", errors="ignore")

        #编码为整数TokenID组成的list
        return tokenizer.encode(text)


if __name__ == "__main__":
    input_file = r"D:\project\cs336\data\TinyStoriesV2-GPT4-train.txt"
    output_file = r"D:\project\cs336\data\TinyStoriesV2-GPT4-train.npy"
    num_processes = 8

    with open(input_file,"rb") as f:
        # 按 <|endoftext|> 边界切分文件，计算各进程的字节偏移区间
        boundaries = find_chunk_boundaries(f, num_processes,b"<|endoftext|>")
        tasks = [(input_file,start,end)for start,end in zip(boundaries[:-1], boundaries[1:])]

        with multiprocessing.Pool(num_processes) as pool:
            results = pool.starmap(process_chunk, tasks)
            #将二维嵌套list转成一维list
            results = [token_id for chunk in results for token_id in chunk]

            #转np.uint16的numpy数组
            all_tokens = np.array(results, dtype =np.uint16)

            np.save(output_file,all_tokens)

            
