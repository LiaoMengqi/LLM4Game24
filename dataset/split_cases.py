import json

with open("24_case.json", "r") as f:
    cases = json.load(f)

train_cases = cases[:-100]
test_cases = cases[-100:]

with open("train_cases.json", "w") as f:
    json.dump(train_cases, f)

with open("test_cases.json", "w") as f:
    json.dump(test_cases, f)
