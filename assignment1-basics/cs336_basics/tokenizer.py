
import regex as re
from typing import Iterable,Iterator
import pickle

GPT2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_PAT =re.compile(GPT2_SPLIT_REGEX)

class Tokenizer:

    def __init__(self,vocab,merges,special_tokens =None):
        self.vocab = vocab
        self.merges =merges
        self.special_tokens = special_tokens
        if self.special_tokens :
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[len(vocab)] = token_bytes
        else:
            self.special_pattern =None

        #反转词表
        self.inverse_vocab ={}
        #元组解包
        for token_id ,token_bytes in self.vocab.items():
            self.inverse_vocab[token_bytes] =token_id
            
        #反转merges
        self.mergedict = {}

        for rank,merge in enumerate(merges):
            self.mergedict[merge]= rank
        #把special_tokens的列表处理成special_pattern的or表达式
        if self.special_tokens:
            escaped_tokens =[]
            #正则表达式要把最长的排在前面，避免被重复拆分 
            sorted_tokens = sorted(self.special_tokens,key=len,reverse =True)
            for token in sorted_tokens:
                escaped_tokens.append(re.escape(token))
            joined_str="|".join(escaped_tokens)
            self.special_pattern = f"({joined_str})"


    def decode(self,ids:list[int] )-> str:
        bt=[]
        for id in ids:
            if id in self.vocab:
                bt.append(self.vocab[id])
            else:
                bt.append(bytes([id]))
        #list不能直接decode，以及用b""做分隔符
        return b"".join(bt).decode("utf-8",errors = "replace")


    def encode(self,text:str) ->list[int]:
        if self.special_pattern:
            parts = re.split(self.special_pattern,text)
        else:
            parts = [text]
        token_ids = []
        for part in parts:
            #空字符处理
            if part == "":
                continue
            #special_tokens处理,要先check special_tokens有东西
            elif self.special_tokens and part in self.special_tokens:
                id = self.inverse_vocab[part.encode("utf-8")]
                token_ids.append(id)
            #正常文本处理
            else:
                tokenlist = re.findall(GPT2_PAT,part)


                for word in tokenlist:
                    raw_bytes = word.encode("utf-8")
                    byte_list = [bytes([b]) for b in raw_bytes]
                
                    while True:
                        if len(byte_list) < 2:
                            break
                        pairs= [(b1,b2) for b1,b2 in zip(byte_list[:-1],byte_list[1:]) if (b1,b2) in self.mergedict]

                        if not pairs:
                            break
                        best_pair = min(pairs,key=lambda p : self.mergedict[p])
                        new_byte_list = []
                        i = 0
                        while i < len(byte_list):
                            if i < len(byte_list) -1 and byte_list[i] == best_pair[0] and byte_list[i + 1] ==best_pair[1]:
                                new_byte_list.append(best_pair[0] + best_pair[1])
                                i +=2

                            else:
                                new_byte_list.append(byte_list[i])
                                i += 1

                        byte_list = new_byte_list
                    for b in byte_list:
                        id = self.inverse_vocab[b]
                        token_ids.append(id)
        return token_ids
    
    
    #encode迭代器
    def encode_iterable(self,iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            idlist = self.encode(line)
            for id in idlist:
                yield id
    
    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath, special_tokens= None):
        with open(vocab_filepath,"rb") as f :
            vocab = pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)


            return cls(vocab,merges,special_tokens)