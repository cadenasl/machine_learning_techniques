import os
import requests
import re
import simpleTokenizer
import simpletokenizerv2

if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print(result)

result = re.split(r'([,.]|\s)', text)
# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])


vocab = {token:integer for integer,token in enumerate(all_tokens)}
vocab_size = len(all_tokens)


print(vocab_size)
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
    
# vocab = simpleTokenizer.SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know," 
#            Mrs. Gisburn said with pardonable pride."""
# ids = vocab.encode(text)
# print(ids)
# print(vocab.decode(ids))

vocab2=simpletokenizerv2.SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)
print(vocab2.encode(text))
print(vocab2.decode(vocab2.encode(text)))
