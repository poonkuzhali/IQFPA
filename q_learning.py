import math
import time
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pygame

symbols = {
    0: '↑',
    1: '→',
    2: '↓',
    3: '←',
    4: 'S',
    5: 'X',
    6: 'T',
    -1: '.'
}


class Grid:
    def __init__(self, row, column, actions, grid, start, goal, obstacles):
        self.n_row = row
        self.n_column = column
        self.n_states = row * column
        self.n_actions = actions
        self.grid = grid
        self.q_table = np.zeros((self.n_row, self.n_column, self.n_actions))
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.rewards = self.get_rewards()

    def get_rewards(self):
        rewards = np.full((self.n_row, self.n_column), -0.04)
        rewards[self.goal[0], self.goal[1]] = 10
        for i in range(self.n_row):
            for j in range(self.n_column):
                if (i, j) in self.obstacles:
                    rewards[i, j] = -10

        return rewards

    # put it inside q learning itself
    def get_action(self, i, j, epsilon=1.0):
        if np.random.rand() < epsilon:
            return np.argmax(self.q_table[i, j])
        else:
            return np.random.randint(self.n_actions)

    def get_next_state(self, row, col, action):
        if action == 0:
            next_state = [max(row - 1, 0), col]
        elif action == 1:
            next_state = [row, min(col + 1, self.n_column - 1)]
        elif action == 2:
            next_state = [min(row + 1, self.n_row - 1), col]
        else:
            next_state = [row, max(col - 1, 0)]

        return next_state[0], next_state[1]

    def calculate_q_value(self, row, column, a, new_r, new_col):
        learning_rate = 0.2
        gamma = 0.8
        reward = self.rewards[new_r, new_col]
        current_q = self.q_table[row, column, a]
        q = current_q + (learning_rate * reward + (gamma * np.max(self.q_table[new_r, new_col])) - current_q)
        return q

    def q_learning(self):
        epsilon = 0.9
        for i in range(200):
            # row, column = np.random.randint(self.n_row), np.random.randint(self.n_column)
            row, column = self.start[0], self.start[1]
            while (row, column) != self.goal:
                a = self.get_action(row, column, epsilon)
                new_r, new_col = self.get_next_state(row, column, a)
                reward = self.rewards[new_r, new_col]
                learning_rate = 0.2
                gamma = 0.8
                current_q = self.q_table[row, column, a]
                q = current_q + (learning_rate * reward + (gamma * np.max(self.q_table[new_r, new_col])) - current_q)
                self.q_table[row, column, a] = q

                row = new_r
                column = new_col

    def get_policy(self):
        policy_grid = np.full((self.n_row, self.n_column), -1)
        row, column = self.start[0], self.start[1]
        policy = [[row, column]]
        while (row, column) != self.goal:
            a = self.get_action(row, column)
            policy_grid[row, column] = a
            row, column = self.get_next_state(row, column, a)
            policy.append((row, column))

        # print(policy)

        # policy_grid[self.start[0], self.start[1]] = 4
        policy_grid[self.goal[0], self.goal[1]] = 6
        for x in self.obstacles:
            policy_grid[x[0], x[1]] = 5
        policy_matrix = np.vectorize(symbols.get)(policy_grid)
        # print(np.array_str(policy_matrix, max_line_width=200))
        return policy


class FPA(Grid):
    def __init__(self, row, column, actions, grid, start, goal, obstacles):
        super(FPA, self).__init__(row, column, actions, grid, start, goal, obstacles)
        self.population_size = 10
        self.lower = 0
        self.upper = 20
        self.grid_size = 20

    def initialize_population(self):
        population = np.zeros((self.population_size, 2), dtype=int)
        for i in range(self.population_size):
            while True:
                x = np.random.randint(self.lower, self.upper)
                y = np.random.randint(self.lower, self.upper)

                if (0 <= x < self.n_row and 0 <= y < self.n_column and (x, y) not in self.obstacles
                        and (x, y) not in self.goal):
                    population[i, 0] = x
                    population[i, 1] = y
                    break

        return population

    def levy_flight(self):
        lambda_ = 1.5
        sigma = (math.gamma(1 + lambda_) * math.sin(math.pi * lambda_ / 2) /
                 (math.gamma((1 + lambda_) / 2) * lambda_ * 2 ** ((lambda_ - 1) / 2))) ** (1 / lambda_)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        s = u / abs(v) ** (1 / lambda_)
        return s

    def global_pollination(self, x, y, best_solution):
        # print(x, y)
        gamma_ = 0.1
        levy_ = self.levy_flight()
        x_new = np.clip(x + (gamma_ * levy_) * (x - best_solution[0]), 0, self.grid_size - 1)
        y_new = np.clip(y + (gamma_ * levy_) * (y - best_solution[1]), 0, self.grid_size - 1)
        return int(x_new), int(y_new)

    def local_pollination(self, x, y, population, current_index):
        # print(x, y)
        indices = np.delete(np.arange(self.population_size), current_index)
        flower1 = np.random.choice(indices)
        flower2 = np.random.choice(indices)
        while flower1 == flower2:
            flower1 = np.random.choice(indices)
        eps = np.random.uniform()
        x_new = np.clip(x + eps * (population[flower1][0] - population[flower2][0]), 0, self.grid_size - 1)
        y_new = np.clip(y + eps * (population[flower1][1] - population[flower2][1]), 0, self.grid_size - 1)
        return int(x_new), int(y_new)

    def evaluate_q_value(self, row, col):
        up = [max(row - 1, 0), col]
        right = [row, min(col + 1, self.n_column - 1)]
        down = [min(row + 1, self.n_row - 1), col]
        left = [row, max(col - 1, 0)]

        self.q_table[row, col, 0] = super().calculate_q_value(row, col, 0, up[0], up[1])
        self.q_table[row, col, 1] = super().calculate_q_value(row, col, 1, right[0], right[1])
        self.q_table[row, col, 2] = super().calculate_q_value(row, col, 2, down[0], down[1])
        self.q_table[row, col, 3] = super().calculate_q_value(row, col, 3, left[0], left[1])

    def fpa_algorithm(self):
        population = self.initialize_population()
        switch_probability = 0.8
        t = 0
        g_best = ()
        g_q_val = float('-inf')
        for x in population:
            if (x[0], x[1]) not in self.obstacles:
                self.evaluate_q_value(x[0], x[1])
                max_q = np.max(self.q_table[x[0], x[1]])
                if max_q > g_q_val:
                    g_q_val = max_q
                    g_best = (x[0], x[1])

        while t < 1000:
            # print(f'Iteration {t}')
            for i in range(self.population_size):
                x, y = population[i]
                if np.random.rand() > switch_probability:
                    x_new, y_new = self.global_pollination(x, y, g_best)
                else:
                    x_new, y_new = self.local_pollination(x, y, population, i)

                self.evaluate_q_value(x_new, y_new)
                q = np.max(self.q_table[x_new, y_new])
                if q > g_q_val:
                    g_q_val = q
                    g_best = (x_new, y_new)

            t = t + 1

        # print(self.q_table)


def display_grids_pygame(grid, start, goal, path1=None, path2=None, obstacles=None):
    # Initialize Pygame
    pygame.init()

    # Set up the window
    cell_size = 30
    grid_rows, grid_cols = grid.shape
    window_width = grid_cols * cell_size * 2 + cell_size  # Add space between the grids
    window_height = grid_rows * cell_size
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Grid with Different Paths")

    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # # Variables to store obstacles
    # obstacles = []

    # Main loop
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get the clicked cell coordinates
                col = event.pos[0] // cell_size
                row = event.pos[1] // cell_size

                # Adjust the column if clicked on the second grid
                if col >= grid_cols + 1:
                    col -= grid_cols + 1

                # Set or remove the obstacle based on the clicked cell
                if event.button == 1:  # Left mouse button
                    if (row, col) not in obstacles:
                        obstacles.append((row, col))
                        grid[row, col] = 1
                elif event.button == 3:  # Right mouse button
                    if (row, col) in obstacles:
                        obstacles.remove((row, col))
                        grid[row, col] = 0

        # Clear the window
        window.fill(WHITE)

        # Draw the grids side by side
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell_rect1 = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                cell_rect2 = pygame.Rect((col + grid_cols + 1) * cell_size, row * cell_size, cell_size, cell_size)

                if (row, col) in obstacles:
                    pygame.draw.rect(window, BLACK, cell_rect1)
                    pygame.draw.rect(window, BLACK, cell_rect2)
                else:
                    pygame.draw.rect(window, WHITE, cell_rect1)
                    pygame.draw.rect(window, WHITE, cell_rect2)

                pygame.draw.rect(window, BLACK, cell_rect1, 1)
                pygame.draw.rect(window, BLACK, cell_rect2, 1)

        start_rect = pygame.Rect(start[1] * cell_size, start[0] * cell_size, cell_size, cell_size)
        goal_rect = pygame.Rect(goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size)
        start_rect2 = pygame.Rect((start[1] + grid_cols + 1) * cell_size, start[0] * cell_size, cell_size, cell_size)
        goal_rect2 = pygame.Rect((goal[1] + grid_cols + 1) * cell_size, goal[0] * cell_size, cell_size, cell_size)
        pygame.draw.ellipse(window, GREEN, start_rect)
        pygame.draw.ellipse(window, BLUE, goal_rect)
        pygame.draw.ellipse(window, GREEN, start_rect2)
        pygame.draw.ellipse(window, BLUE, goal_rect2)

        # Draw the paths if available
        if path1 is not None:
            for i in range(len(path1) - 1):
                x1, y1 = path1[i]
                x2, y2 = path1[i + 1]
                pygame.draw.line(window, RED, (y1 * cell_size + cell_size // 2, x1 * cell_size + cell_size // 2),
                                 (y2 * cell_size + cell_size // 2, x2 * cell_size + cell_size // 2), 3)

        if path2 is not None:
            for i in range(len(path2) - 1):
                x1, y1 = path2[i]
                x2, y2 = path2[i + 1]
                pygame.draw.line(window, BLUE,
                                 ((y1 + grid_cols + 1) * cell_size + cell_size // 2, x1 * cell_size + cell_size // 2),
                                 ((y2 + grid_cols + 1) * cell_size + cell_size // 2, x2 * cell_size + cell_size // 2),
                                 3)

        # Update the display
        pygame.display.update()

    # Quit Pygame
    pygame.quit()

    return obstacles


def calculate_total_traveled_distance(policy):
    total = 0
    for i in range(len(policy) - 1):
        total += math.dist(policy[i], policy[i + 1])

    return total


def calculate_path_smoothness(policy):
    total = 0
    for i in range(len(policy) - 1):
        x1, y1 = policy[i - 1]
        x2, y2 = policy[i]
        x3, y3 = policy[i + 1]
        angle1 = math.atan2(y2 - y1, x2 - x1)
        angle2 = math.atan2(y3 - y2, x3 - x2)

        total += abs(angle2 - angle1)

    return total


if __name__ == '__main__':
    obstacles = [(2, 6), (3, 5), (3, 6), (3, 7), (3, 8), (2, 7), (1, 6), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
                 (8, 11), (8, 13), (8, 12), (8, 10), (9, 11), (9, 12), (10, 11), (10, 12), (7, 17), (7, 18), (6, 18),
                 (6, 17), (7, 16), (5, 18), (18, 13), (17, 13), (16, 13), (17, 14), (18, 14), (16, 15), (15, 15),
                 (17, 15), (17, 16), (18, 16), (18, 17), (8, 0), (8, 1), (8, 2), (8, 3), (9, 2), (9, 1), (14, 3),
                 (14, 4), (14, 5), (15, 4), (15, 2), (15, 3), (17, 3), (17, 4), (17, 6), (17, 7), (17, 5), (18, 4),
                 (18, 5), (18, 6), (14, 9), (14, 10), (14, 11), (13, 10), (12, 10), (12, 9), (12, 8), (11, 8), (11, 7)]

    grid = np.zeros((20, 20))
    # obstacle = display_grid_pygame2(grid,(4, 2), (9, 18))
    # print('Obstacles')
    # print(obstacle)

    fpa1 = FPA(20, 20, 4, grid, (4, 4), (9, 18), obstacles)
    start_time = time.time()
    fpa1.q_learning()
    print(f'Time for Q {time.time() - start_time}')
    # print(np.array_str(fpa.q_table, max_line_width=200))
    policy1 = fpa1.get_policy()
    print(calculate_total_traveled_distance(policy1))

    print(f'Path smoothness {calculate_path_smoothness(policy1)}')
    # print(fpa.get_policy())
    # display_grid_pygame2(fpa.grid,(4, 2), (9, 18), policy1)

    grid = np.zeros((20, 20))
    fpa2 = FPA(20, 20, 4, grid, (4, 4), (9, 18), obstacles)
    start_time = time.time()
    fpa2.fpa_algorithm()
    fpa2.q_learning()
    print(f'Time for IQFPA {time.time() - start_time}')
    # print(np.array_str(fpa.q_table, max_line_width=200))
    policy2 = fpa2.get_policy()
    print(calculate_total_traveled_distance(policy2))


    print(f'Path smoothness {calculate_path_smoothness(policy2)}')

    # display_grid_pygame2(fpa.grid,(4, 2), (9, 18), policy2)
    display_grids_pygame(grid, (4, 4), (9, 18), policy1, policy2, obstacles=obstacles)
