import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

integers = tokenizer.encode("Akwirw ier")
print(integers)

for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")
print(tokenizer.encode("w"))

print(tokenizer.encode("ir"))

print(tokenizer.encode("w"))

print(tokenizer.encode(" "))

print(tokenizer.encode("ier"))

print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))

