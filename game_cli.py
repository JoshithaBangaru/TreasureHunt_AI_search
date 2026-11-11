#!/usr/bin/env python3
"""
Interactive BFS vs UCS Pathfinding Simulator & Benchmark
Zero External Dependencies (Pure Python standard library: heapq, collections, time, itertools).
Run immediately with:
    python3 game_cli.py
"""

import sys
import time
import heapq
from collections import deque
import itertools
import random

# ANSI Colors for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"
BG_GREEN = "\033[42m\033[30m"
BG_RED = "\033[41m\033[37m"
BG_YELLOW = "\033[43m\033[30m"

TERRAIN_COSTS = {
    '.': 1,  # Grass
    '~': 3,  # Mud
    '=': 5,  # Water
    '#': float('inf'),  # Obstacle
}

TERRAIN_NAMES = {
    '.': "Grass (Cost: 1)",
    '~': "Mud (Cost: 3)",
    '=': "Water (Cost: 5)",
    '#': "Obstacle (Blocked)",
}

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def bfs_search(grid, start, goal):
    """Breadth-First Search finding minimal step count."""
    t_start = time.perf_counter()
    rows, cols = len(grid), len(grid[0])
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
                if grid[nr][nc] != '#' and (nr, nc) not in visited:
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

    total_cost = sum(TERRAIN_COSTS[grid[r][c]] for r, c in path[1:])
    return {
        "path": path,
        "cost": total_cost,
        "steps": len(path) - 1,
        "nodes_expanded": nodes_expanded,
        "time_ms": exec_time_ms,
        "reachable": True
    }

def ucs_search(grid, start, goal):
    """Uniform Cost Search minimizing total cumulative movement cost."""
    t_start = time.perf_counter()
    rows, cols = len(grid), len(grid[0])
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
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                step_cost = TERRAIN_COSTS[grid[nr][nc]]
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

def solve_tsp(grid, start, goal, treasures, algo="UCS"):
    """Multi-goal treasure tour visiting all treasures then goal."""
    search_fn = ucs_search if algo == "UCS" else bfs_search
    t_start = time.perf_counter()

    if not treasures:
        res = search_fn(grid, start, goal)
        return {**res, "treasures_collected": 0, "treasure_points": 0, "net_score": -res["cost"] if res["reachable"] else 0}

    points = [start] + [t["pos"] for t in treasures] + [goal]
    n_treasures = len(treasures)
    treasure_indices = list(range(1, n_treasures + 1))

    memo = {}
    def get_segment(i, j):
        if (i, j) not in memo:
            memo[(i, j)] = search_fn(grid, points[i], points[j])
        return memo[(i, j)]

    best_order = None
    best_metric = float('inf')

    for perm in itertools.permutations(treasure_indices):
        seq = [0] + list(perm) + [len(points) - 1]
        valid = True
        metric = 0
        for idx in range(len(seq) - 1):
            leg = get_segment(seq[idx], seq[idx+1])
            if not leg["reachable"]:
                valid = False
                break
            metric += leg["steps"] if algo == "BFS" else leg["cost"]
        if valid and metric < best_metric:
            best_metric = metric
            best_order = seq

    t_end = time.perf_counter()
    total_time = max(0.05, (t_end - t_start) * 1000.0)

    if not best_order:
        return {"path": [], "cost": 0, "steps": 0, "nodes_expanded": 0, "time_ms": total_time, "reachable": False, "net_score": 0}

    full_path = []
    tot_cost = 0
    tot_steps = 0
    tot_nodes = 0
    for idx in range(len(best_order) - 1):
        leg = get_segment(best_order[idx], best_order[idx+1])
        tot_cost += leg["cost"]
        tot_steps += leg["steps"]
        tot_nodes += leg["nodes_expanded"]
        if not full_path:
            full_path.extend(leg["path"])
        else:
            full_path.extend(leg["path"][1:])

    tot_pts = sum(t["value"] for t in treasures)
    net_score = tot_pts - tot_cost

    return {
        "path": full_path,
        "cost": tot_cost,
        "steps": tot_steps,
        "nodes_expanded": tot_nodes,
        "time_ms": total_time,
        "reachable": True,
        "treasures_collected": n_treasures,
        "treasure_points": tot_pts,
        "net_score": net_score
    }

def print_grid(grid, start, goal, treasures, bfs_path, ucs_path):
    """Render terminal grid with ANSI color styling."""
    bfs_set = set(bfs_path)
    ucs_set = set(ucs_path)
    t_map = {t["pos"]: t["value"] for t in treasures}

    print("\n   " + "".join(f"{c:2d}" for c in range(len(grid[0]))))
    print("  +" + "--" * len(grid[0]) + "+")

    for r in range(len(grid)):
        row_str = f"{r:2d}|"
        for c in range(len(grid[0])):
            pos = (r, c)
            cell = grid[r][c]

            if pos == start:
                row_str += f"{BG_GREEN} S{RESET}"
            elif pos == goal:
                row_str += f"{BG_RED} G{RESET}"
            elif pos in t_map:
                row_str += f"{BG_YELLOW} T{RESET}"
            elif pos in bfs_set and pos in ucs_set:
                row_str += f"{MAGENTA} *{RESET}"
            elif pos in bfs_set:
                row_str += f"{RED} B{RESET}"
            elif pos in ucs_set:
                row_str += f"{BLUE} U{RESET}"
            elif cell == '#':
                row_str += f"{GRAY}██{RESET}"
            elif cell == '=':
                row_str += f"{CYAN}~~{RESET}"
            elif cell == '~':
                row_str += f"{YELLOW}░░{RESET}"
            else:
                row_str += f"{GREEN}..{RESET}"
        row_str += "|"
        print(row_str)
    print("  +" + "--" * len(grid[0]) + "+")

def get_preset(choice):
    if choice == "1":
        # Easy Meadow (8x8)
        size = 8
        grid = [['.' for _ in range(size)] for _ in range(size)]
        grid[2][2] = grid[2][3] = grid[3][2] = grid[3][3] = '#'
        grid[4][1] = grid[4][2] = grid[5][1] = grid[5][2] = '~'
        start = (0, 0)
        goal = (7, 7)
        treasures = [{"pos": (3, 4), "value": 35}]
        return grid, start, goal, treasures, "Easy · Meadow Crossing (8x8)"
    elif choice == "3":
        # Hard Archipelago (12x12)
        size = 12
        grid = [['.' for _ in range(size)] for _ in range(size)]
        for c in range(size):
            grid[5][c] = grid[6][c] = '='
        grid[5][2] = grid[6][2] = '.'  # Bridge 1
        grid[5][9] = grid[6][9] = '.'  # Bridge 2
        grid[3][3] = grid[3][4] = grid[8][8] = grid[8][9] = '#'
        start = (1, 1)
        goal = (10, 10)
        treasures = [
            {"pos": (1, 9), "value": 60},
            {"pos": (6, 6), "value": 100},
            {"pos": (9, 2), "value": 75}
        ]
        return grid, start, goal, treasures, "Hard · Archipelago & Straits (12x12)"
    else:
        # Medium Swamp Trap (10x10) - Default
        size = 10
        grid = [['.' for _ in range(size)] for _ in range(size)]
        for r in range(3, 7):
            for c in range(3, 7):
                grid[r][c] = '~'
        for r in range(4, 6):
            for c in range(4, 6):
                grid[r][c] = '='
        grid[3][2] = grid[4][2] = grid[5][7] = grid[6][7] = '#'
        start = (1, 1)
        goal = (8, 8)
        treasures = [
            {"pos": (2, 7), "value": 50},
            {"pos": (7, 2), "value": 60}
        ]
        return grid, start, goal, treasures, "Medium · The Swamp Shortcut Trap (10x10)"

def main():
    print(f"\n{BOLD}{CYAN}======================================================{RESET}")
    print(f"{BOLD}{CYAN}   🧭 BFS vs UCS Pathfinding Simulator & Audit       {RESET}")
    print(f"{BOLD}{CYAN}======================================================{RESET}")
    print(f"{DIM}Comparing Step-Hop Optimization (BFS) vs Movement Cost Optimization (UCS){RESET}\n")

    print(f"{BOLD}Select Map Scenario:{RESET}")
    print("  [1] Easy · Meadow Crossing (8x8)")
    print("  [2] Medium · The Swamp Shortcut Trap (10x10) [Default]")
    print("  [3] Hard · Archipelago & Golden Straits (12x12)")
    print("  [4] Random Procedural Terrain (10x10)")

    try:
        choice = input(f"\nEnter choice [1-4] (default=2): ").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        choice = "2"

    if choice == "4":
        size = 10
        grid = [['.' for _ in range(size)] for _ in range(size)]
        start = (0, 0)
        goal = (size - 1, size - 1)
        treasures = [{"pos": (size // 2, size // 2), "value": 50}]
        rng = random.Random(42)
        for r in range(size):
            for c in range(size):
                if (r, c) in [start, goal, treasures[0]["pos"]]:
                    continue
                v = rng.random()
                if v < 0.12:
                    grid[r][c] = '#'
                elif v < 0.25:
                    grid[r][c] = '='
                elif v < 0.45:
                    grid[r][c] = '~'
        map_title = "Procedural Terrain (10x10)"
    else:
        grid, start, goal, treasures, map_title = get_preset(choice)

    print(f"\n{BOLD}Scenario Loaded:{RESET} {CYAN}{map_title}{RESET}")
    print(f"Start: {GREEN}{start}{RESET} | Goal: {RED}{goal}{RESET} | Treasures: {YELLOW}{len(treasures)}{RESET}")

    # Run Algorithms
    print(f"\n{DIM}Computing optimal paths...{RESET}")
    bfs_res = solve_tsp(grid, start, goal, treasures, algo="BFS")
    ucs_res = solve_tsp(grid, start, goal, treasures, algo="UCS")

    # Render Grid
    print_grid(grid, start, goal, treasures, bfs_res["path"], ucs_res["path"])

    print(f"\n{BOLD}Legend:{RESET} {BG_GREEN}S{RESET} Start  {BG_RED}G{RESET} Goal  {BG_YELLOW}T{RESET} Treasure  {RED}B{RESET} BFS Path  {BLUE}U{RESET} UCS Path  {MAGENTA}*{RESET} Overlap  {GREEN}..{RESET} Grass(1)  {YELLOW}░░{RESET} Mud(3)  {CYAN}~~{RESET} Water(5)  {GRAY}██{RESET} Wall")

    # Comparative Metrics Table
    print(f"\n{BOLD}================ PERFORMANCE AUDIT ================{RESET}")
    print(f"{'Metric':<25} | {'🔴 BFS (Hops)':<18} | {'🔵 UCS (Least Cost)':<18}")
    print("-" * 68)
    print(f"{'Total Movement Cost':<25} | {bfs_res['cost']:<18} | {BOLD}{BLUE}{ucs_res['cost']}{RESET}{' (Optimal)':<8}")
    print(f"{'Total Steps (Hops)':<25} | {BOLD}{RED}{bfs_res['steps']}{RESET}{' (Minimal)':<8} | {ucs_res['steps']:<18}")
    print(f"{'Nodes Expanded':<25} | {bfs_res['nodes_expanded']:<18} | {ucs_res['nodes_expanded']:<18}")
    print(f"{'Execution Time (ms)':<25} | {bfs_res['time_ms']:.2f} ms{'':<11} | {ucs_res['time_ms']:.2f} ms")
    print(f"{'Net Score (Reward-Cost)':<25} | {bfs_res['net_score']} pts{'':<10} | {BOLD}{GREEN}{ucs_res['net_score']} pts{RESET}")
    print("=" * 68)

    # Automated Narrative Verdict
    cost_diff = bfs_res['cost'] - ucs_res['cost']
    steps_diff = ucs_res['steps'] - bfs_res['steps']

    print(f"\n{BOLD}⚖️  AUTOMATED VERDICT:{RESET}")
    if cost_diff > 0:
        pct = (cost_diff / bfs_res['cost']) * 100
        print(f"  {GREEN}✓ UCS OUTPERFORMED BFS ON MOVEMENT COST:{RESET}")
        print(f"    UCS saved {cost_diff} movement cost units ({pct:.1f}% reduction).")
        print(f"    While BFS greedily followed the shortest geometric hops ({bfs_res['steps']} steps),")
        print(f"    it was severely penalized by crossing high-friction Mud (cost=3) or Water (cost=5).")
        print(f"    UCS deliberately took {steps_diff} extra step(s) through Grass (cost=1) to achieve")
        print(f"    the mathematically optimal path cost ({ucs_res['cost']} vs {bfs_res['cost']}).")
        print(f"    Net Score advantage: +{ucs_res['net_score'] - bfs_res['net_score']} points.")
    elif cost_diff == 0:
        print(f"  {CYAN}✓ TIE ON TRAVERSAL EXPENDITURE:{RESET}")
        print(f"    Both algorithms converged on the exact same optimal route ({ucs_res['steps']} steps, cost: {ucs_res['cost']}).")
        print(f"    BFS completed with minimal deque overhead.")
    print()

if __name__ == "__main__":
    main()
