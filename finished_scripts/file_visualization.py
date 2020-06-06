import os
import matplotlib.pyplot as plt

locations_path = "data\locations"
locations = os.listdir(locations_path)
locations = sorted(map(int, locations))
locations = list(map(str, locations))

size_array = []
for loc in locations:
    loc_path = os.path.join(locations_path, loc)
    size_array.append(len(os.listdir(loc_path)))


plt.figure(figsize=(20, 5))
plt.grid(axis='y')
plt.bar(locations, size_array, log=True)
plt.xticks(rotation=60, fontsize=7)
plt.autoscale(axis='both', tight=True)
plt.show()