import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 16,
    "font.family": "Times New Roman",
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
    "figure.titlesize": 18
})

random.seed(500)
np.random.seed(500)

G = nx.Graph()

fake_users = [f"FU{i}" for i in range(10)]
real_users = [f"RU{i}" for i in range(30)]
products = [f"PR{i}" for i in range(5)]

G.add_nodes_from(fake_users + real_users + products)
users = fake_users + real_users

for user in users:
    p = random.choice(products)
    G.add_edge(user, p)

pos = {}

left_outlier = real_users[0]
top_outlier = real_users[1]

cluster = fake_users[:8] + real_users[2:22] + products[:4]
upper = fake_users[8:] + real_users[22:26]
outer = real_users[26:] + [products[4]]

center = np.array([0.65, 0.5])
for n in cluster:
    pos[n] = center + np.random.normal(scale=0.11, size=2)

upper_center = np.array([0.7, 0.72])
for n in upper:
    pos[n] = upper_center + np.random.normal(scale=0.07, size=2)

anchors = [(0.5, 0.2), (0.8, 0.2), (0.9, 0.5), (0.55, 0.85), (0.4, 0.4)]
for i, n in enumerate(outer):
    base = np.array(anchors[i % len(anchors)])
    pos[n] = base + np.random.normal(scale=0.03, size=2)

pos[left_outlier] = np.array([0.18, 0.55])
pos[top_outlier] = np.array([0.92, 0.92])

edge_widths = [random.uniform(0.25, 1.2) for _ in G.edges()]
node_sizes_random = {n: random.randint(60, 420) for n in G.nodes()}

def estimate_node_radii(pos, node_sizes, axis_width_in=4.6, x_range=0.96, scale=1.12):
    radii = {}
    data_per_point = x_range / (axis_width_in * 72.0)
    for n, s in node_sizes.items():
        r_points = np.sqrt(s / np.pi)
        radii[n] = r_points * data_per_point * scale
    return radii

def remove_node_overlap(pos, radii, iterations=1500, margin=0.008):
    nodes = list(pos.keys())
    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                delta = pos[n1] - pos[n2]
                dist = np.linalg.norm(delta)
                min_allowed = radii[n1] + radii[n2] + margin
                if dist < min_allowed:
                    moved = True
                    if dist == 0:
                        direction = np.random.randn(2)
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = delta / dist
                    shift = (min_allowed - dist) / 2.0
                    pos[n1] += direction * shift
                    pos[n2] -= direction * shift
        for n in nodes:
            r = radii[n] + margin
            pos[n][0] = np.clip(pos[n][0], 0.02 + r, 0.98 - r)
            pos[n][1] = np.clip(pos[n][1], 0.02 + r, 0.98 - r)
        if not moved:
            break

radii = estimate_node_radii(pos, node_sizes_random, axis_width_in=4.6, x_range=0.96, scale=1.12)
remove_node_overlap(pos, radii, iterations=1500, margin=0.008)

for n in pos:
    pos[n][0] = np.clip(pos[n][0] - 0.06, 0.02 + radii[n], 0.98 - radii[n])

remove_node_overlap(pos, radii, iterations=1000, margin=0.008)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
nx.draw_networkx_edges(G, pos, edge_color="#888888", width=0.45, alpha=0.45, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=fake_users, node_color="#9ecae1", node_size=85, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=real_users, node_color="#f4a6b7", node_size=85, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=products, node_color="#bdbdbd", node_size=105, ax=ax)

ax = axes[1]
nx.draw_networkx_edges(G, pos, edge_color="red", width=edge_widths, alpha=0.6, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=fake_users, node_color="#9ecae1", node_size=85, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=real_users, node_color="#f4a6b7", node_size=85, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=products, node_color="#bdbdbd", node_size=105, ax=ax)

ax = axes[2]
nx.draw_networkx_edges(G, pos, edge_color="red", width=edge_widths, alpha=0.6, ax=ax)
nx.draw_networkx_nodes(
    G, pos, nodelist=fake_users, node_color="#9ecae1",
    node_size=[node_sizes_random[n] for n in fake_users],
    edgecolors='black', linewidths=1.0, ax=ax
)
nx.draw_networkx_nodes(
    G, pos, nodelist=real_users, node_color="#f4a6b7",
    node_size=[node_sizes_random[n] for n in real_users],
    edgecolors='black', linewidths=1.0, ax=ax
)
nx.draw_networkx_nodes(
    G, pos, nodelist=products, node_color="#bdbdbd",
    node_size=[node_sizes_random[n] for n in products],
    edgecolors='black', linewidths=1.1, ax=ax
)

for ax in axes:
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.05, 0.98)
    ax.set_aspect('equal')
    ax.axis("off")

axes[0].text(0.68, -0.01, "(a) MODE-base",
             transform=axes[0].transAxes, ha="center", va="top")
axes[1].text(0.62, -0.01, "(b) MODE-W",
             transform=axes[1].transAxes, ha="center", va="top")
axes[2].text(0.62, -0.01, "(c) MODE-WT",
             transform=axes[2].transAxes, ha="center", va="top")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Fake User',
           markerfacecolor="#9ecae1", markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Real User',
           markerfacecolor="#f4a6b7", markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Product',
           markerfacecolor="#bdbdbd", markersize=8),
    Line2D([0], [0], marker='o', color='black', label='Reconstruction',
           markerfacecolor="white", markersize=8)
]

axes[0].legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(0.00, 1.10),
    frameon=False
)

plt.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.06, wspace=0.015)

plt.savefig("three_graphs_font_adjusted.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()