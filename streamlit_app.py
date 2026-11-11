"""
BFS vs UCS Pathfinding Simulator

Compares Breadth-First Search (fewest steps) with Uniform Cost Search
(lowest total terrain cost) on a grid with grass, mud, water and walls.

Run it with:
    pip install streamlit numpy matplotlib pandas
    streamlit run streamlit_app.py
"""

import heapq
import itertools
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="BFS vs UCS Pathfinding Simulator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide the Deploy button, the three-dot menu (Settings lives in there),
# the header bar, the footer and the sidebar arrow.
st.markdown(
    """
    <style>
        .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"], #MainMenu, header, footer {
            display: none !important;
        }
        .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Terrain codes used in the grid: 0 grass, 1 mud, 2 water, 3 wall
COST = {0: 1, 1: 3, 2: 5}
WALL = 3
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def neighbours(grid, cell):
    r, c = cell
    rows, cols = grid.shape
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != WALL:
            yield (nr, nc)


def build_path(parent, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    return path


def not_found(expanded, ms):
    return {"path": [], "cost": 0, "steps": 0, "nodes_expanded": expanded,
            "time_ms": ms, "reachable": False}


def bfs(grid, start, goal):
    began = time.perf_counter()
    queue = deque([start])
    seen = {start}
    parent = {start: None}
    expanded = 0
    found = False

    while queue:
        cell = queue.popleft()
        expanded += 1
        if cell == goal:
            found = True
            break
        for nxt in neighbours(grid, cell):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = cell
                queue.append(nxt)

    ms = max(0.01, (time.perf_counter() - began) * 1000)
    if not found:
        return not_found(expanded, ms)

    path = build_path(parent, goal)
    return {
        "path": path,
        "cost": sum(COST[grid[r, c]] for r, c in path[1:]),
        "steps": len(path) - 1,
        "nodes_expanded": expanded,
        "time_ms": ms,
        "reachable": True,
    }


def ucs(grid, start, goal):
    began = time.perf_counter()
    heap = [(0, start)]
    best = {start: 0}
    parent = {start: None}
    done = set()
    expanded = 0
    found = False

    while heap:
        cost_here, cell = heapq.heappop(heap)
        if cell in done:
            continue
        done.add(cell)
        expanded += 1
        if cell == goal:
            found = True
            break
        for nxt in neighbours(grid, cell):
            new_cost = cost_here + COST[grid[nxt]]
            if nxt not in best or new_cost < best[nxt]:
                best[nxt] = new_cost
                parent[nxt] = cell
                heapq.heappush(heap, (new_cost, nxt))

    ms = max(0.01, (time.perf_counter() - began) * 1000)
    if not found:
        return not_found(expanded, ms)

    path = build_path(parent, goal)
    return {
        "path": path,
        "cost": best[goal],
        "steps": len(path) - 1,
        "nodes_expanded": expanded,
        "time_ms": ms,
        "reachable": True,
    }


def solve(grid, start, goal, treasures, algo):
    """Try every order of visiting the treasures and keep the cheapest tour."""
    search = ucs if algo == "UCS" else bfs
    began = time.perf_counter()

    if not treasures:
        res = search(grid, start, goal)
        leg = {"from": "Start", "to": "Goal", "path": res["path"],
               "cost": res["cost"], "steps": res["steps"]}
        return {**res, "treasure_points": 0,
                "net_score": -res["cost"] if res["reachable"] else 0,
                "legs": [leg]}

    points = [start] + [t["pos"] for t in treasures] + [goal]
    names = (["Start"]
             + [f"Treasure {i + 1} (+{t['value']} pts)" for i, t in enumerate(treasures)]
             + ["Goal"])

    cache = {}

    def leg_between(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = search(grid, points[i], points[j])
        return cache[(i, j)]

    best_order, best_score = None, float("inf")
    for order in itertools.permutations(range(1, len(treasures) + 1)):
        route = [0, *order, len(points) - 1]
        total = 0
        ok = True
        for a, b in zip(route, route[1:]):
            leg = leg_between(a, b)
            if not leg["reachable"]:
                ok = False
                break
            total += leg["steps"] if algo == "BFS" else leg["cost"]
        if ok and total < best_score:
            best_score, best_order = total, route

    ms = max(0.05, (time.perf_counter() - began) * 1000)
    if best_order is None:
        return {**not_found(0, ms), "treasure_points": 0, "net_score": 0, "legs": []}

    full_path, legs = [], []
    cost = steps = nodes = 0
    for a, b in zip(best_order, best_order[1:]):
        leg = leg_between(a, b)
        cost += leg["cost"]
        steps += leg["steps"]
        nodes += leg["nodes_expanded"]
        legs.append({"from": names[a], "to": names[b], "path": leg["path"],
                     "cost": leg["cost"], "steps": leg["steps"]})
        full_path.extend(leg["path"] if not full_path else leg["path"][1:])

    points_total = sum(t["value"] for t in treasures)
    return {
        "path": full_path,
        "cost": cost,
        "steps": steps,
        "nodes_expanded": nodes,
        "time_ms": ms,
        "reachable": True,
        "treasure_points": points_total,
        "net_score": points_total - cost,
        "legs": legs,
    }


def draw_grid(grid, start, goal, treasures, bfs_path, ucs_path):
    rows, cols = grid.shape
    colours = ListedColormap(["#86efac", "#d97706", "#38bdf8", "#334155"])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(grid, cmap=colours, origin="upper", vmin=0, vmax=3)

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5, alpha=0.5)
    ax.tick_params(which="minor", size=0)

    # nudge the two lines apart a little so they don't hide each other
    if bfs_path:
        r, c = zip(*bfs_path)
        ax.plot([x - 0.08 for x in c], [y - 0.08 for y in r], color="#ef4444",
                linewidth=3.5, linestyle="--", marker="o", markersize=4,
                label="BFS path (fewest steps)")
    if ucs_path:
        r, c = zip(*ucs_path)
        ax.plot([x + 0.08 for x in c], [y + 0.08 for y in r], color="#2563eb",
                linewidth=4, marker="s", markersize=4,
                label="UCS path (lowest cost)")

    ax.scatter(start[1], start[0], color="#16a34a", s=260, zorder=5,
               edgecolors="white", linewidth=2)
    ax.text(start[1], start[0], "S", color="white", weight="bold",
            ha="center", va="center", fontsize=12, zorder=6)

    ax.scatter(goal[1], goal[0], color="#dc2626", s=260, zorder=5, marker="*",
               edgecolors="white", linewidth=2)
    ax.text(goal[1], goal[0], "G", color="white", weight="bold",
            ha="center", va="center", fontsize=11, zorder=6)

    for i, t in enumerate(treasures):
        tr, tc = t["pos"]
        ax.scatter(tc, tr, color="#eab308", s=230, zorder=5, marker="D",
                   edgecolors="#713f12", linewidth=1.5)
        ax.text(tc, tr, f"T{i + 1}", color="#713f12", weight="bold",
                ha="center", va="center", fontsize=9, zorder=6)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout()
    return fig


def preset_map(name):
    """Returns (grid, start, goal, treasures) for one of the built-in levels."""
    if name.startswith("Easy"):
        grid = np.zeros((8, 8), dtype=int)
        grid[2:4, 2:4] = 3
        grid[4:6, 1:3] = 1
        return grid, (0, 0), (7, 7), [{"pos": (3, 4), "value": 35}]

    if name.startswith("Hard"):
        grid = np.zeros((12, 12), dtype=int)
        grid[5:7, :] = 2            # river
        grid[5:7, 2] = 0            # bridge 1
        grid[5:7, 9] = 0            # bridge 2
        grid[3, 3:5] = 3
        grid[8, 8:10] = 3
        treasures = [{"pos": (1, 9), "value": 60},
                     {"pos": (6, 6), "value": 100},
                     {"pos": (9, 2), "value": 75}]
        return grid, (1, 1), (10, 10), treasures

    if name.startswith("Benchmark"):
        grid = np.zeros((10, 10), dtype=int)
        grid[[2, 6], 1:9] = 3       # walls
        grid[3:6, :] = 1            # mud corridor
        grid[7:, :] = 2             # water at the bottom
        return grid, (4, 0), (4, 9), [{"pos": (1, 5), "value": 40}]

    # Medium: the swamp trap
    grid = np.zeros((10, 10), dtype=int)
    grid[3:7, 3:7] = 1
    grid[4:6, 4:6] = 2
    grid[3, 2] = grid[4, 2] = 3
    grid[5, 7] = grid[6, 7] = 3
    treasures = [{"pos": (2, 7), "value": 50}, {"pos": (7, 2), "value": 60}]
    return grid, (1, 1), (8, 8), treasures


def random_map(size, mud, water, walls, seed):
    grid = np.zeros((size, size), dtype=int)
    start, goal = (0, 0), (size - 1, size - 1)
    treasures = [{"pos": (size // 2, size // 2), "value": 50}]
    keep_clear = {start, goal, treasures[0]["pos"]}

    rng = np.random.default_rng(seed)
    for r in range(size):
        for c in range(size):
            if (r, c) in keep_clear:
                continue
            roll = rng.random()
            if roll < walls:
                grid[r, c] = 3
            elif roll < walls + water:
                grid[r, c] = 2
            elif roll < walls + water + mud:
                grid[r, c] = 1
    return grid, start, goal, treasures


# ---------------------------------------------------------------- page

st.title("🧭 BFS vs UCS Pathfinding Simulator")
st.markdown(
    "Compare **Breadth-First Search** (fewest steps) with **Uniform Cost Search** "
    "(lowest total movement cost) on a grid with different terrain."
)

# Options now live on the page itself instead of the sidebar
with st.container(border=True):
    st.subheader("Map & Environment Setup")

    top_left, top_right = st.columns([1, 1])
    with top_left:
        mode = st.radio("Map creation mode",
                        ["Predefined Levels", "Custom Map Builder"],
                        horizontal=True)
    with top_right:
        collect_treasures = st.checkbox(
            "Treasure hunt: visit all treasures before the goal", value=True)

    if mode == "Predefined Levels":
        level = st.selectbox("Preset level", [
            "Medium: The Swamp Shortcut Trap (10x10)",
            "Easy: Meadow Crossing (8x8)",
            "Hard: Archipelago & Straits (12x12)",
            "Benchmark: The Triple Corridor (10x10)",
        ])
        grid, start, goal, treasures = preset_map(level)
    else:
        a, b, c, d, e = st.columns(5)
        size = a.slider("Grid size (N x N)", 6, 14, 10)
        mud = b.slider("Mud density", 0.0, 0.4, 0.15)
        water = c.slider("Water density", 0.0, 0.4, 0.10)
        walls = d.slider("Obstacle density", 0.0, 0.3, 0.10)
        seed = e.number_input("Random seed", value=42, step=1)
        grid, start, goal, treasures = random_map(size, mud, water, walls, seed)

active_treasures = treasures if collect_treasures else []
bfs_res = solve(grid, start, goal, active_treasures, "BFS")
ucs_res = solve(grid, start, goal, active_treasures, "UCS")

col_map, col_stats = st.columns([1.1, 1])

with col_map:
    st.subheader("Grid & Paths")
    st.pyplot(draw_grid(grid, start, goal, active_treasures,
                        bfs_res["path"], ucs_res["path"]))
    st.caption("Grass (cost 1, green) | Mud (cost 3, orange) | "
               "Water (cost 5, blue) | Wall (blocked, dark)")

with col_stats:
    st.subheader("Performance Audit")
    left, right = st.columns(2)

    with left:
        st.markdown("### 🔴 BFS")
        st.metric("Total path cost", bfs_res["cost"])
        st.metric("Steps (hops)", bfs_res["steps"])
        st.metric("Nodes expanded", bfs_res["nodes_expanded"])
        st.metric("Execution time", f"{bfs_res['time_ms']:.2f} ms")
        if collect_treasures:
            st.metric("Net score", f"{bfs_res['net_score']} pts")

    with right:
        st.markdown("### 🔵 UCS")
        cost_delta = ucs_res["cost"] - bfs_res["cost"]
        st.metric("Total path cost", ucs_res["cost"],
                  delta=f"{cost_delta} cost" if bfs_res["cost"] else None,
                  delta_color="inverse")
        st.metric("Steps (hops)", ucs_res["steps"],
                  delta=f"{ucs_res['steps'] - bfs_res['steps']} hops")
        st.metric("Nodes expanded", ucs_res["nodes_expanded"])
        st.metric("Execution time", f"{ucs_res['time_ms']:.2f} ms")
        if collect_treasures:
            st.metric("Net score", f"{ucs_res['net_score']} pts",
                      delta=f"{ucs_res['net_score'] - bfs_res['net_score']} pts")

# Verdict
st.markdown("---")
st.subheader("Verdict")

cost_saved = bfs_res["cost"] - ucs_res["cost"]
extra_steps = ucs_res["steps"] - bfs_res["steps"]

if not (bfs_res["reachable"] and ucs_res["reachable"]):
    st.error("There is no valid path from start to goal with these obstacles.")
elif cost_saved > 0:
    pct = cost_saved / bfs_res["cost"] * 100
    st.success(
        f"**UCS beat BFS on movement cost.** It saved **{cost_saved} cost units** "
        f"({pct:.1f}% less). BFS took the shortest route in hops "
        f"({bfs_res['steps']} steps) but crossed expensive mud or water. "
        f"UCS took **{extra_steps} extra step(s)** over cheaper ground instead. "
        f"Net score difference: **+{ucs_res['net_score'] - bfs_res['net_score']} points**."
    )
elif cost_saved == 0 and extra_steps == 0:
    st.info(
        f"**Tie.** Both ended up on the same path ({ucs_res['steps']} steps, "
        f"cost {ucs_res['cost']}). There was no cheaper detour on this map."
    )
else:
    st.info(f"Both found a path. UCS cost: {ucs_res['cost']}, BFS cost: {bfs_res['cost']}.")

# Per-leg breakdown
if collect_treasures and ucs_res["legs"]:
    st.markdown("---")
    st.subheader("Leg-by-leg breakdown")
    left, right = st.columns(2)
    for col, label, res in ((left, "BFS legs", bfs_res), (right, "UCS legs", ucs_res)):
        with col:
            st.markdown(f"**{label}**")
            table = pd.DataFrame([
                {"From": l["from"], "To": l["to"], "Steps": l["steps"], "Cost": l["cost"]}
                for l in res["legs"]
            ])
            st.dataframe(table, width="stretch")
