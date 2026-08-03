import time 
import tracemalloc 
from bpe import train_bpe
import pickle
def main():
    tracemalloc.start()
    start_time = time.time()

    vocab,merges = train_bpe(
        input_path = "D:/project/cs336/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size = 10000,
        special_tokens=["<|endoftext|>"]
    )

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with open("tinystories_vocab.pkl","wb") as f:
        pickle.dump(vocab,f)


    with open("tinystories_merges.pkl","wb") as f:
        pickle.dump(merges,f)
        


    print(f"--- 训练完成 ---")
    print(f"耗时 (Time): {elapsed_time:.2f} 秒")
    print(f"内存峰值 (Peak Memory): {peak / (1024 ** 3):.2f} GB")

    longest_token = max(vocab.values(), key=len)
    print(f"最长的 Token: {repr(longest_token)}")
    print(f"最长 Token 的长度: {len(longest_token)}")

if __name__ == "__main__":
    main()