from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os 
from tqdm import tqdm
GPT2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_PAT =re.compile(GPT2_SPLIT_REGEX)

def process_chunk(file_path:str, start:int ,end:int,special_tokens: list[str]) ->Counter:
    counts = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start).replace(b"\r\n", b"\n")
        chunk_text = chunk_bytes.decode("utf-8", errors="replace")

    if special_tokens:
        pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        sub_chunks = [c for c in re.split(pattern, chunk_text) if c and c not in special_tokens]
    else:
        sub_chunks = [chunk_text]

    for sub in sub_chunks:
        for match in GPT2_PAT.finditer(sub):
            raw_bytes = match.group(0).encode("utf-8")
            byte_tuple= tuple(bytes([b]) for b in raw_bytes)
            counts[byte_tuple] += 1
    return counts






def train_bpe(input_path:str ,
                vocab_size: int ,
                special_tokens: list[str]
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:

    #基础的256个字节代表
    vocab ={}
    for  i in range(256):
        vocab[i] = bytes([i])


    #存放specialtokens
    if special_tokens:
        for tok in special_tokens:
            vocab[len(vocab)] = tok.encode("utf-8")

 
    #求出要merges多少次
    num_merges = vocab_size - 256-len(special_tokens)
    if num_merges <= 0:
        return vocab,[]

    num_processes =  4

    #取第一个specialtoken先分块，多线程处理每个块
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(f,num_processes ,split_token)


    counts = Counter()
    with ProcessPoolExecutor(max_workers =num_processes ) as executor:
        futures =[
            executor.submit(process_chunk,str(input_path),start,end,special_tokens)
            #滑动窗口，双指针写法
            for start,end in zip(boundaries[:-1],boundaries[1:])
        ]
        for future in futures:
            counts.update(future.result())
    #全局管理pair_counts，节省时间
    pair_counts = Counter()
    for word_tuple,freq in counts.items():
        if len(word_tuple) < 2:
            continue
        for i in range(len(word_tuple) -1 ):
            pair = (word_tuple[i],word_tuple[i + 1])
            pair_counts[pair] += freq


    merges: list[tuple[bytes,bytes]] =[]


    #合并查找
    for _ in tqdm(range(num_merges)):

        if not pair_counts:
            break

        best_pair = max(pair_counts,key=lambda p: (pair_counts[p], p))
        new_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        vocab[len(vocab)] = new_token

    
        
        new_counts = Counter()
        for word_tuple, freq in counts.items():
            if best_pair[0] not in word_tuple or len(word_tuple) < 2:
                new_counts[word_tuple] += freq
                continue
            new_word=[]
            i= 0
            n = len(word_tuple)
            while i < n:
                if i < n -1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] ==best_pair[1]:
                    #同步更新全局的paircount表
                    if i> 0:
                        pair_counts[(word_tuple[i -1 ],word_tuple[i])] -= freq
                    if i+ 2< n:
                        pair_counts[(word_tuple[i + 1],word_tuple[i+2])] -= freq

                    if i > 0:
                        pair_counts[(word_tuple[i -1 ],new_token)] += freq
                    if i+ 2< n:
                        pair_counts[(new_token,word_tuple[i+2])] += freq

                    new_word.append(new_token)
                    i +=2
                        
                else:
                    new_word.append(word_tuple[i])
                    i += 1


            pair_counts[best_pair] -= freq
            if pair_counts[best_pair] <= 0:
                del pair_counts[best_pair]
            new_counts[tuple(new_word)] += freq
        
        counts = new_counts


    return vocab,merges









    



