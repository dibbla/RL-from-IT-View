import heapq

def astar(start, goal, grid):
    # Manhattan distance
    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

    # Initialize open and closed sets
    open_set = []
    closed_set = set()

    # Add start node to open set
    heapq.heappush(open_set, (0, start))

    # Initialize g score and f score dictionaries
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # Initialize came_from dictionary
    came_from = {}

    # Loop until open set is empty
    while open_set:
        # Pop node with lowest f score from open set
        current = heapq.heappop(open_set)[1]

        # Check if current node is goal
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                parent = came_from[current]
                move = (current[0] - parent[0], current[1] - parent[1])
                path.append((current, move))
                current = parent
            path.append((start, None))
            path.reverse()
            return path

        # Add current node to closed set
        closed_set.add(current)

        # Loop through neighbors of current node
        for neighbor in [(current[0]+1, current[1]), (current[0]-1, current[1]), (current[0], current[1]+1), (current[0], current[1]-1)]:
            # Check if neighbor is in grid and not a wall
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] != 2:
                # Calculate tentative g score
                tentative_g_score = g_score[current] + 1

                # Check if neighbor is already in closed set or if tentative g score is greater than existing g score
                if neighbor in closed_set or tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                # Update came_from, g score, and f score
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                # Add neighbor to open set
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # If goal is not reachable, return None
    return None