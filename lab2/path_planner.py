from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """

    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        # all nodes are already initialized with f n g as INF
        pq = []
        cost_map = self.cost_map
        grid = self.node_grid
        start = grid.get_node(start_position[0], start_position[1])
        start.g = 0
        goal = grid.get_node(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start.g, start))
        while pq:
            _, node = heapq.heappop(pq)
            if node.closed:
                continue
            node.closed = True
            node_pos = node.get_position()
            if node_pos == goal_position:
                break
            for successor_pos in grid.get_successors(node_pos[0], node_pos[1]):
                successor = grid.get_node(successor_pos[0], successor_pos[1])
                if successor.closed:
                    continue
                cost = cost_map.get_edge_cost(node_pos, successor_pos)
                if successor.g > node.g + cost:
                    successor.g = node.g + cost
                    successor.parent = node
                    heapq.heappush(pq, (successor.g, successor))

        path = self.construct_path(goal)
        cost_g = goal.g
        self.node_grid.reset()
        return path, cost_g

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # all nodes are already initialized with f n g as INF
        pq = []
        cost_map = self.cost_map
        grid = self.node_grid
        start = grid.get_node(start_position[0], start_position[1])
        start.g = 0
        start.f = start.distance_to(goal_position[0], goal_position[1])
        goal = grid.get_node(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start.f, start))
        while pq:
            _, node = heapq.heappop(pq)
            if node.closed:
                continue
            node.closed = True
            node_pos = node.get_position()
            if node_pos == goal_position:
                break
            for successor_pos in grid.get_successors(node_pos[0], node_pos[1]):
                successor = grid.get_node(successor_pos[0], successor_pos[1])
                if successor.closed:
                    continue
                cost = cost_map.get_edge_cost(node_pos, successor_pos)
                if successor.g > node.g + cost:
                    successor.g = node.g + cost
                    successor.f = successor.distance_to(
                        goal_position[0], goal_position[1]
                    )
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))

        path = self.construct_path(goal)
        cost_g = goal.g
        self.node_grid.reset()
        return path, cost_g

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # all nodes are already initialized with f n g as INF
        pq = []
        cost_map = self.cost_map
        grid = self.node_grid
        start = grid.get_node(start_position[0], start_position[1])
        start.g = 0
        start.f = start.distance_to(goal_position[0], goal_position[1])
        goal = grid.get_node(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start.f, start))
        while pq:
            _, node = heapq.heappop(pq)
            if node.closed:
                continue
            node.closed = True
            node_pos = node.get_position()
            if node_pos == goal_position:
                break
            for successor_pos in grid.get_successors(node_pos[0], node_pos[1]):
                successor = grid.get_node(successor_pos[0], successor_pos[1])
                if successor.closed:
                    continue
                cost = cost_map.get_edge_cost(node_pos, successor_pos)
                if successor.g > node.g + cost:
                    successor.g = node.g + cost
                    successor.f = successor.g + successor.distance_to(
                        goal_position[0], goal_position[1]
                    )
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))

        path = self.construct_path(goal)
        cost_g = goal.g
        self.node_grid.reset()
        return path, cost_g
