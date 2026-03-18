with open("/home/student/Documents/ooais-project/data/processed/final.csv") as f:
	lines = f.readlines()

data = [line.split(",") for line in lines[1:]]
print("Number of records:", len(lines)-1)
objects = data[1]
print("Objects:", set(objects))


avg_temperature = sum(data[2]) / len(data)
print(f"{avg_temperature=}")

lst = [line.split(",")[1] for line in lines[1:]]
counts = {}
for item in lst:
    counts[item] = counts.get(item, 0) + 1

print(counts)
