import json

with open('train_data.json', 'r') as f:
    data = json.load(f)

dataset = []
for item in data:
    temp = dict()
    item = item.split('\n')
    temp["input"] = item[0] + '\n'
    temp["output"] = "\n".join(item[1:])
    dataset.append(temp)

with open('train.json', 'w') as f:
    print(len(dataset))
    json.dump(dataset, f)

with open('mini_train.json', 'w') as f:
    json.dump(dataset[:100], f)