"""
Streamlit Web Application: BFS vs UCS Pathfinding Simulator & Audit
Technical Stack: Python 3.8+, Streamlit, NumPy, Matplotlib, heapq, collections.deque

QUICKSTART GUIDE:
-----------------
Option 1: Run Locally on your computer
    1. pip install streamlit numpy matplotlib
    2. streamlit run streamlit_app.py

Option 2: Run in Google Colab (Free browser runtime)
    In a new Colab notebook, paste and run:
    !pip install streamlit pyngrok
    # Save this file as streamlit_app.py
    # Run with localtunnel or ngrok
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import heapq
from collections import deque
import itertools
import time
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="BFS vs UCS Pathfinding Simulator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cost Constants
TERRAIN_COSTS = {
    0: 1,  # Grass
    1: 3,  # Mud
    2: 5,  # Water
    3: float('inf')  # Obstacle
}

TERRAIN_LABELS = {
    0: "Grass (Cost: 1)",
    1: "Mud (Cost: 3)",
    2: "Water (Cost: 5)",
    3: "Obstacle (Blocked)"
}

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def bfs_single(grid, start, goal):
    """
    Breadth-First Search (BFS) finding path with minimal step count.
    Uses collections.deque FIFO queue.
    """
    t_start = time.perf_counter()
    rows, cols = grid.shape
    queue = deque([start])
    visited = {start}
    parent = {}
    nodes_expanded = 0
    reached = False

    while queue:
        curr = queue.popleft()
        nodes_expanded += 1

        if curr == goal:
            reached = True
            break

        r, c = curr
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != 3 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = curr
                    queue.append((nr, nc))

    t_end = time.perf_counter()
    exec_time_ms = max(0.01, (t_end - t_start) * 1000.0)

    if not reached:
        return {"path": [], "cost": 0, "steps": 0, "nodes_expanded": nodes_expanded, "time_ms": exec_time_ms, "reachable": False}

    path = []
    curr = goal
    while curr:
        path.append(curr)
        curr = parent.get(curr)
    path.reverse()

    total_cost = sum(TERRAIN_COSTS[grid[r, c]] for r, c in path[1:])
    total_steps = len(path) - 1

    return {
        "path": path,
        "cost": total_cost,
        "steps": total_steps,
        "nodes_expanded": nodes_expanded,
        "time_ms": exec_time_ms,
        "reachable": True
    }

def ucs_single(grid, start, goal):
    """
    Uniform Cost Search (UCS) minimizing total movement cost.
    Uses heapq priority queue keyed by cumulative g(n).
    """
    t_start = time.perf_counter()
    rows, cols = grid.shape
    pq = [(0, start)]
    cost_so_far = {start: 0}
    parent = {}
    visited = set()
    nodes_expanded = 0
    reached = False

    while pq:
        curr_cost, curr = heapq.heappop(pq)

        if curr in visited:
            continue
        visited.add(curr)
        nodes_expanded += 1

        if curr == goal:
            reached = True
            break

        r, c = curr
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 3:
                step_cost = TERRAIN_COSTS[grid[nr, nc]]
                new_cost = curr_cost + step_cost
                if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                    cost_so_far[(nr, nc)] = new_cost
                    parent[(nr, nc)] = curr
                    heapq.heappush(pq, (new_cost, (nr, nc)))

    t_end = time.perf_counter()
    exec_time_ms = max(0.01, (t_end - t_start) * 1000.0)

    if not reached:
        return {"path": [], "cost": 0, "steps": 0, "nodes_expanded": nodes_expanded, "time_ms": exec_time_ms, "reachable": False}

    path = []
    curr = goal
    while curr:
        path.append(curr)
        curr = parent.get(curr)
    path.reverse()

    return {
        "path": path,
        "cost": cost_so_far[goal],
        "steps": len(path) - 1,
        "nodes_expanded": nodes_expanded,
        "time_ms": exec_time_ms,
        "reachable": True
    }

def solve_multi_goal(grid, start, goal, treasures, algo="UCS"):
    """
    Computes optimal multi-goal tour visiting all treasures then reaching goal.
    """
    fn = ucs_single if algo == "UCS" else bfs_single
    t_start = time.perf_counter()

    if not treasures:
        res = fn(grid, start, goal)
        total_pts = 0
        net_score = -res["cost"] if res["reachable"] else 0
        return {**res, "treasures_collected": 0, "treasure_points": 0, "net_score": net_score, "legs": [{"from": "Start", "to": "Goal", "path": res["path"], "cost": res["cost"], "steps": res["steps"]}]}

    points = [start] + [t["pos"] for t in treasures] + [goal]
    point_names = ["Start"] + [f"Treasure {i+1} (+{t['value']} pts)" for i, t in enumerate(treasures)] + ["Goal"]
    n_treasures = len(treasures)
    treasure_indices = list(range(1, n_treasures + 1))

    # Precalculate leg segments
    segment_cache = {}
    def get_leg(i, j):
        if (i, j) not in segment_cache:
            segment_cache[(i, j)] = fn(grid, points[i], points[j])
        return segment_cache[(i, j)]

    # Best permutation
    best_order = None
    best_metric = float('inf')

    perms = list(itertools.permutations(treasure_indices))
    for p in perms:
        seq = [0] + list(p) + [len(points) - 1]
        valid = True
        metric = 0
        for idx in range(len(seq) - 1):
            leg = get_leg(seq[idx], seq[idx+1])
            if not leg["reachable"]:
                valid = False
                break
            metric += leg["steps"] if algo == "BFS" else leg["cost"]
        if valid and metric < best_metric:
            best_metric = metric
            best_order = seq

    t_end = time.perf_counter()
    total_time_ms = max(0.05, (t_end - t_start) * 1000.0)

    if not best_order:
        return {"path": [], "cost": 0, "steps": 0, "nodes_expanded": 0, "time_ms": total_time_ms, "reachable": False, "net_score": 0, "treasures_collected": 0, "treasure_points": 0, "legs": []}

    # Stitch full path
    full_path = []
    tot_cost = 0
    tot_steps = 0
    tot_nodes = 0
    legs = []

    for idx in range(len(best_order) - 1):
        u, v = best_order[idx], best_order[idx+1]
        leg = get_leg(u, v)
        tot_cost += leg["cost"]
        tot_steps += leg["steps"]
        tot_nodes += leg["nodes_expanded"]
        legs.append({
            "from": point_names[u],
            "to": point_names[v],
            "path": leg["path"],
            "cost": leg["cost"],
            "steps": leg["steps"]
        })
        if not full_path:
            full_path.extend(leg["path"])
        else:
            full_path.extend(leg["path"][1:])

    tot_treasure_pts = sum(t["value"] for t in treasures)
    net_score = tot_treasure_pts - tot_cost

    return {
        "path": full_path,
        "cost": tot_cost,
        "steps": tot_steps,
        "nodes_expanded": tot_nodes,
        "time_ms": total_time_ms,
        "reachable": True,
        "treasures_collected": n_treasures,
        "treasure_points": tot_treasure_pts,
        "net_score": net_score,
        "legs": legs
    }

def render_matplotlib_grid(grid, start, goal, treasures, bfs_path, ucs_path):
    """
    Renders high-contrast Matplotlib visual grid.
    """
    rows, cols = grid.shape
    # Custom colormap: Grass=emerald, Mud=amber-brown, Water=cerulean, Obstacle=charcoal
    cmap = ListedColormap(['#86efac', '#d97706', '#38bdf8', '#334155'])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(grid, cmap=cmap, origin='upper', vmin=0, vmax=3)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.tick_params(which="minor", size=0)

    # Plot BFS path
    if bfs_path:
        br, bc = zip(*bfs_path)
        ax.plot([c - 0.08 for c in bc], [r - 0.08 for r in br], color='#ef4444', linewidth=3.5, label='BFS Path (Hop-Optimal)', linestyle='--', marker='o', markersize=4)

    # Plot UCS path
    if ucs_path:
        ur, uc = zip(*ucs_path)
        ax.plot([c + 0.08 for c in uc], [r + 0.08 for r in ur], color='#2563eb', linewidth=4, label='UCS Path (Cost-Optimal)', marker='s', markersize=4)

    # Start and Goal markers
    ax.scatter(start[1], start[0], color='#16a34a', s=260, zorder=5, marker='o', edgecolors='white', linewidth=2)
    ax.text(start[1], start[0], 'S', color='white', weight='bold', ha='center', va='center', fontsize=12, zorder=6)

    ax.scatter(goal[1], goal[0], color='#dc2626', s=260, zorder=5, marker='*', edgecolors='white', linewidth=2)
    ax.text(goal[1], goal[0], 'G', color='white', weight='bold', ha='center', va='center', fontsize=11, zorder=6)

    # Treasures
    for idx, t in enumerate(treasures):
        tr, tc = t["pos"]
        ax.scatter(tc, tr, color='#eab308', s=230, zorder=5, marker='D', edgecolors='#713f12', linewidth=1.5)
        ax.text(tc, tr, f'T{idx+1}', color='#713f12', weight='bold', ha='center', va='center', fontsize=9, zorder=6)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout()
    return fig

# --- Main App Interface ---
st.title("🧭 BFS vs UCS Pathfinding Simulator & Audit")
st.markdown("Simulate and compare **Breadth-First Search (BFS)** (step-hop optimization) and **Uniform Cost Search (UCS)** (cumulative movement cost minimization) on a customizable weighted terrain grid.")

# Sidebar Configuration
st.sidebar.header("Map & Environment Setup")
mode = st.sidebar.radio("Map Creation Mode", ["Predefined Levels", "Custom Map Builder"])

if mode == "Predefined Levels":
    level_choice = st.sidebar.selectbox("Select Preset Level", [
        "Medium: The Swamp Shortcut Trap (10x10)",
        "Easy: Meadow Crossing (8x8)",
        "Hard: Archipelago & Straits (12x12)",
        "Benchmark: The Triple Corridor (10x10)"
    ])

    if "Easy" in level_choice:
        size = 8
        grid = np.zeros((size, size), dtype=int)
        grid[2:4, 2:4] = 3  # obstacle
        grid[4:6, 1:3] = 1  # mud
        start = (0, 0)
        goal = (7, 7)
        treasures = [{"pos": (3, 4), "value": 35}]
    elif "Hard" in level_choice:
        size = 12
        grid = np.zeros((size, size), dtype=int)
        grid[5:7, :] = 2  # river water
        grid[5:7, 2] = 0  # bridge 1
        grid[5:7, 9] = 0  # bridge 2
        grid[3, 3:5] = 3  # obstacles
        grid[8, 8:10] = 3
        start = (1, 1)
        goal = (10, 10)
        treasures = [
            {"pos": (1, 9), "value": 60},
            {"pos": (6, 6), "value": 100},
            {"pos": (9, 2), "value": 75}
        ]
    elif "Triple Corridor" in level_choice:
        size = 10
        grid = np.zeros((size, size), dtype=int)
        start = (4, 0)
        goal = (4, 9)
        treasures = [{"pos": (1, 5), "value": 40}]
        grid[[2, 6], 1:9] = 3  # walls
        grid[3:6, :] = 1       # middle mud
        grid[7:, :] = 2        # bottom water
    else:
        # Medium Swamp Trap
        size = 10
        grid = np.zeros((size, size), dtype=int)
        grid[3:7, 3:7] = 1  # mud
        grid[4:6, 4:6] = 2  # water
        grid[3, 2] = 3
        grid[4, 2] = 3
        grid[5, 7] = 3
        grid[6, 7] = 3
        start = (1, 1)
        goal = (8, 8)
        treasures = [
            {"pos": (2, 7), "value": 50},
            {"pos": (7, 2), "value": 60}
        ]
else:
    size = st.sidebar.slider("Grid Dimension (N x N)", 6, 14, 10)
    grid = np.zeros((size, size), dtype=int)
    start = (0, 0)
    goal = (size - 1, size - 1)
    treasures = [{"pos": (size // 2, size // 2), "value": 50}]

    st.sidebar.markdown("**Procedural Terrain Sliders:**")
    mud_prob = st.sidebar.slider("Mud Density", 0.0, 0.4, 0.15)
    water_prob = st.sidebar.slider("Water Density", 0.0, 0.4, 0.1)
    obstacle_prob = st.sidebar.slider("Obstacle Density", 0.0, 0.3, 0.1)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    rng = np.random.default_rng(seed)
    for r in range(size):
        for c in range(size):
            if (r, c) == start or (r, c) == goal or (r, c) == treasures[0]["pos"]:
                continue
            rand_val = rng.random()
            if rand_val < obstacle_prob:
                grid[r, c] = 3
            elif rand_val < obstacle_prob + water_prob:
                grid[r, c] = 2
            elif rand_val < obstacle_prob + water_prob + mud_prob:
                grid[r, c] = 1

collect_treasures = st.sidebar.checkbox("Treasure Hunt: Visit All Treasures Before Goal", value=True)

# Run algorithms
bfs_res = solve_multi_goal(grid, start, goal, treasures if collect_treasures else [], algo="BFS")
ucs_res = solve_multi_goal(grid, start, goal, treasures if collect_treasures else [], algo="UCS")

# Top Layout: Visualization & Metrics
col_map, col_metrics = st.columns([1.1, 1])

with col_map:
    st.subheader("Grid Matrix & Trajectories")
    fig = render_matplotlib_grid(grid, start, goal, treasures if collect_treasures else [], bfs_res["path"], ucs_res["path"])
    st.pyplot(fig)

    st.caption("Terrain Weights: Grass (cost: 1, green) | Mud (cost: 3, orange) | Water (cost: 5, blue) | Wall (blocked, slate)")

with col_metrics:
    st.subheader("Performance Audit")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("### 🔴 BFS (Breadth-First)")
        st.metric("Total Path Cost", f"{bfs_res['cost']}")
        st.metric("Step Count (Hops)", f"{bfs_res['steps']}")
        st.metric("Nodes Expanded", f"{bfs_res['nodes_expanded']}")
        st.metric("Execution Time", f"{bfs_res['time_ms']:.2f} ms")
        if collect_treasures:
            st.metric("Net Score", f"{bfs_res['net_score']} pts")

    with mcol2:
        st.markdown("### 🔵 UCS (Uniform Cost)")
        cost_delta = f"{ucs_res['cost'] - bfs_res['cost']} cost" if bfs_res['cost'] != 0 else None
        st.metric("Total Path Cost", f"{ucs_res['cost']}", delta=cost_delta, delta_color="inverse")
        step_delta = f"{ucs_res['steps'] - bfs_res['steps']} hops"
        st.metric("Step Count (Hops)", f"{ucs_res['steps']}", delta=step_delta)
        st.metric("Nodes Expanded", f"{ucs_res['nodes_expanded']}")
        st.metric("Execution Time", f"{ucs_res['time_ms']:.2f} ms")
        if collect_treasures:
            score_delta = f"+{ucs_res['net_score'] - bfs_res['net_score']} pts"
            st.metric("Net Score", f"{ucs_res['net_score']} pts", delta=score_delta)

# Automated Verdict Narrative
st.markdown("---")
st.subheader("Automated Verdict & Algorithmic Audit")

cost_diff = bfs_res['cost'] - ucs_res['cost']
steps_diff = ucs_res['steps'] - bfs_res['steps']

if not bfs_res["reachable"] or not ucs_res["reachable"]:
    st.error("No valid path exists between start and goal through the obstacles.")
elif cost_diff > 0:
    pct = (cost_diff / bfs_res['cost']) * 100
    st.success(
        f"**UCS Outperformed BFS in Movement Cost**: UCS saved **{cost_diff} cost units** ({pct:.1f}% reduction). "
        f"While BFS selected the shortest geometric hop sequence ({bfs_res['steps']} steps), "
        f"it traversed high-friction Mud (cost=3) or Water (cost=5). "
        f"UCS deliberately took **{steps_diff} extra step(s)** through Grass (cost=1) to minimize total traversal expenditure. "
        f"Net Score advantage: **+{ucs_res['net_score'] - bfs_res['net_score']} points**."
    )
elif cost_diff == 0 and steps_diff == 0:
    st.info(
        f"**Tie**: Both BFS and UCS converged on the exact same optimal path ({ucs_res['steps']} steps, cost: {ucs_res['cost']}). "
        f"The topology offered no cheaper detour around the direct path."
    )
else:
    st.info(f"Both algorithms identified paths. UCS Cost: {ucs_res['cost']}, BFS Cost: {bfs_res['cost']}.")

# Multi-Segment Breakdown Table
if collect_treasures and ucs_res["legs"]:
    st.markdown("---")
    st.subheader("Multi-Goal Leg Details")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("**BFS Legs:**")
        df_bfs = pd.DataFrame([{"From": l["from"], "To": l["to"], "Steps": l["steps"], "Cost": l["cost"]} for l in bfs_res["legs"]])
        st.dataframe(df_bfs, use_container_width=True)
    with col_l2:
        st.markdown("**UCS Legs:**")
        df_ucs = pd.DataFrame([{"From": l["from"], "To": l["to"], "Steps": l["steps"], "Cost": l["cost"]} for l in ucs_res["legs"]])
        st.dataframe(df_ucs, use_container_width=True)
