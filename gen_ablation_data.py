import copy
import json

from transformers import AutoTokenizer

model_name = "d:/model/Qwen/Qwen2___5-0___5B-Instruct/"
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('dataset/medium_data.json', 'r') as f:
    medium = json.load(f)

format_v2 = []
v2_maxlen = 0
for item in medium:
    item = copy.deepcopy(item)
    output = item['output'].split("\n")
    new_output = ""
    for line in output:
        if "expression" in line:
            new_output += line
        else:
            line = line.split(":")
            numbers = line[1].split(",")
            new_numbers = []
            for number in numbers:
                if "=" in number:
                    new_numbers.append(number.split("=")[-1].strip())
                else:
                    new_numbers.append(number.strip())
            new_numbers = ", ".join(new_numbers)
            new_output += line[0] + ": " + new_numbers + "\n"
    item["output"] = new_output
    v2_maxlen = max(v2_maxlen, len(tokenizer.encode(item["input"] + item["output"])))
    format_v2.append(item)

format_v1 = []
v1_maxlen = 0
for item in format_v2:
    item = copy.deepcopy(item)
    output = item['output'].split("\n")
    new_output = ""
    for line in output:

        if "roll" not in line:
            if "expression" in line:
                new_output += line
            else:
                new_output += line + "\n"
    item["output"] = new_output
    v1_maxlen = max(v1_maxlen, len(tokenizer.encode(item["input"] + item["output"])))
    format_v1.append(item)

print("format v1")
print(format_v1[0]["input"])
print(format_v1[0]["output"])
print("----------------------------")
print("format v2")
print(format_v1[0]["input"])
print(format_v2[0]["output"])
print("----------------------------")
print("format v3")
print(medium[0]["input"])
print(medium[0]["output"])

print(v1_maxlen, v2_maxlen)

with open('dataset/format_v1.json', 'w') as f:
    json.dump(format_v1, f)

with open('dataset/format_v2.json', 'w') as f:
    json.dump(format_v2, f)
