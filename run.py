import os
from apriori import Apriori
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

PATH = os.getcwd() + "/test/data/T10I4D100K.dat"
SUPPORT = 6
CONFIDENCE = 0.75
N = 300

# create a new instance of Apriori
apriori = Apriori(PATH, n=N, sep=' ', duplicates=False, verbose=True)

for basket in apriori.baskets[:10]:
    print(basket)

# get candidates and frequent itemsets
candidates, frequent_itemsets = apriori.compute(support=SUPPORT, verbose=False)

print("Frequent items:")
for x in frequent_itemsets[-1]:
    print(list(x))

# get association rules
association_rules = apriori.get_association_rules(confidence=CONFIDENCE, verbose=False)

print("Association rules:")
for r, (c, s) in association_rules.items():
    print("Rule: {} -> {} - Confidence: {} - Support: {}".format(list(r[0]), list(r[1]), c, s))

# scatterplot
df = pd.DataFrame(association_rules.values())
df.columns = ["confidence", "support"]
df.plot.scatter(x="confidence", y="support")


# graph
plt.figure(figsize=(10, 10))
G = nx.DiGraph()
for r, (c, s) in association_rules.items():
    end = list(r[1])[0]
    for rx in list(r[0]):
        G.add_edge(rx, end, weight=1, arrowsize=100)
edges = [
    (u, v) for (u, v, d) in G.edges(data=True)
]
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=1000)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
plt.axis("off")
plt.show()
