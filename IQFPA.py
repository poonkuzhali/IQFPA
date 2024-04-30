import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    def __init__(self, row, column, actions, grid, start, goal, obstacles, max_iter):
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
        self.max_iter = max_iter

    def get_rewards(self):
        rewards = np.full((self.n_row, self.n_column), -0.04)
        rewards[self.goal[0], self.goal[1]] = 10
        for i in range(self.n_row):
            for j in range(self.n_column):
                if (i, j) in self.obstacles:
                    rewards[i, j] = -10

        return rewards

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
        lr = 0.2
        gamma = 0.8
        reward = self.rewards[new_r, new_col]
        current_q = self.q_table[row, column, a]
        q = current_q + (lr * reward + (gamma * np.max(self.q_table[new_r, new_col])) - current_q)
        return q

    def q_learning(self):
        epsilon = 0.8
        for i in range(self.max_iter):
            row, column = self.start[0], self.start[1]
            while (row, column) != self.goal:
                a = self.get_action(row, column, epsilon)
                new_r, new_col = self.get_next_state(row, column, a)
                reward = self.rewards[new_r, new_col]
                lr = 0.2
                gamma = 0.8
                current_q = self.q_table[row, column, a]
                q = current_q + (lr * reward + (gamma * np.max(self.q_table[new_r, new_col])) - current_q)
                self.q_table[row, column, a] = q

                row = new_r
                column = new_col

    def plot_grid_world(self, name, obstacle_idx, iteration, path=None):
        plt.figure(figsize=(8, 8))
        plt.title(f'{name}_{iteration}')
        plt.imshow(np.zeros((21, 21)), cmap='twilight_r')

        if obstacle_idx == 2:
            obstacle_lines = [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
                 (12, 12)],
                [(14, 4), (15, 5), (16, 6), (17, 7), (18, 8), (19, 9)],
                [(4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19)]]
            for obs in obstacle_lines:
                obstacle_x, obstacle_y = zip(*obs)
                plt.plot(obstacle_x, obstacle_y, 'k-', linewidth=2)
        elif obstacle_idx == 3:
            obstacle_lines = [[(0, 14), (1, 14), (2, 14), (3, 14)], [(7, 6), (7,7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12)],
                        [(7, 10), (8, 10), (9, 10), (11, 10), (12, 10), (13, 10), (14, 10)], [(10, 0), (10, 1), (10, 2),
                        (10, 3), (10, 4), (10, 5), (10, 6)], [(12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19)],
                        [(14, 4), (14, 5), (14, 6)], [(14, 4), (15, 4), (16, 4), (17, 4), (18, 4), (19, 4)]]
            for obs in obstacle_lines:
                obstacle_x, obstacle_y = zip(*obs)
                plt.plot(obstacle_x, obstacle_y, 'k-', linewidth=2)
        else:
            for obstacle in self.obstacles:
                plt.plot(obstacle[0], obstacle[1], 'ks', markersize=20)

        plt.plot(self.start[0], self.start[1], 'ro', markersize=10)
        plt.plot(self.goal[0], self.goal[1], 'go', markersize=10)

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, 'b-', linewidth=2)

        plt.xlim(0, 19)
        plt.ylim(0, 19)
        custom_ticks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        plt.xticks(custom_ticks)
        plt.yticks(custom_ticks)
        plt.grid(True)
        plt.savefig(f'{name}/{iteration}.png')
        plt.close()

    def get_policy(self):
        policy_grid = np.full((self.n_row, self.n_column), -1)
        row, column = self.start[0], self.start[1]
        policy = [[row, column]]
        while (row, column) != self.goal:
            a = self.get_action(row, column)
            policy_grid[row, column] = a
            row, column = self.get_next_state(row, column, a)
            policy.append((row, column))

        policy_grid[self.goal[0], self.goal[1]] = 6
        for x in self.obstacles:
            policy_grid[x[0], x[1]] = 5
        # policy_matrix = np.vectorize(symbols.get)(policy_grid)
        return policy


class FPA(Grid):
    def __init__(self, row, column, actions, grid, start, goal, obstacles, population_size, max_iter):
        super(FPA, self).__init__(row, column, actions, grid, start, goal, obstacles, max_iter)
        self.population_size = population_size
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
        gamma_ = 0.1
        levy_ = self.levy_flight()
        x_new = np.clip(x + (gamma_ * levy_) * (x - best_solution[0]), 0, self.grid_size - 1)
        y_new = np.clip(y + (gamma_ * levy_) * (y - best_solution[1]), 0, self.grid_size - 1)
        return int(x_new), int(y_new)

    def local_pollination(self, x, y, population, current_index):
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


def calculate_total_traveled_distance(policy):
    total = 0
    for i in range(len(policy) - 1):
        total += math.dist(policy[i + 1], policy[i])

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


def calculate_percentage_improvement(old, new):
    return round(((old - new) / old) * 100, 2) if old != 0 else 0


def initialize_and_process(obstacles_list, population_size):
    q_learning_max_iter = 200
    average_time_taken_q = []
    average_traveled_distance_q = []
    average_path_smoothness_q = []
    std_time_taken_q = []
    std_traveled_distance_q = []
    std_path_smoothness_q = []
    average_time_taken = []
    average_traveled_distance = []
    average_path_smoothness = []
    std_time_taken = []
    std_traveled_distance = []
    std_path_smoothness = []

    start = (2, 16)
    goal = (19, 11)

    for idx, obstacles in enumerate(obstacles_list):
        print(f'Test case {idx}')
        total_time_q = []
        path_distance_q = []
        smoothness_q = []

        for i in range(30):
            print(f'Q-learning iteration {i}')
            grid = np.zeros((20, 20))
            q_learning = FPA(20, 20, 4, grid, start, goal, obstacles, population_size, q_learning_max_iter)

            # Calculating time - Q learning
            start_time = time.time()
            q_learning.q_learning()
            total_time_q.append(time.time() - start_time)

            policy = q_learning.get_policy()
            path_distance_q.append(calculate_total_traveled_distance(policy))
            smoothness_q.append(calculate_path_smoothness(policy))
            q_learning.plot_grid_world(f'Q-learning_{idx + 1}', idx, i, policy)

        average_time_taken_q.append(round(np.average(total_time_q), 2))
        average_path_smoothness_q.append(round(np.average(smoothness_q), 2))
        average_traveled_distance_q.append(round(np.average(path_distance_q), 2))
        std_time_taken_q.append(round(np.std(total_time_q), 2))
        std_path_smoothness_q.append(round(np.std(smoothness_q), 2))
        std_traveled_distance_q.append(round(np.std(path_distance_q), 2))

        total_time = []
        path_distance = []
        smoothness = []
        for i in range(30):
            print(f'IQFPA iteration {i}')
            grid = np.zeros((20, 20))
            fpa = FPA(20, 20, 4, grid, start, goal, obstacles, population_size, q_learning_max_iter)

            # Calculating time - FPA + Q learning
            start_time = time.time()
            fpa.fpa_algorithm()
            fpa.q_learning()
            total_time.append(time.time() - start_time)

            policy = fpa.get_policy()
            path_distance.append(calculate_total_traveled_distance(policy))
            smoothness.append(calculate_path_smoothness(policy))
            fpa.plot_grid_world(f'IQFPA_{idx + 1}', idx, i, policy)

        average_time_taken.append(round(np.average(total_time), 2))
        average_path_smoothness.append(round(np.average(smoothness), 2))
        average_traveled_distance.append(round(np.average(path_distance), 2))
        std_time_taken.append(round(np.std(total_time), 2))
        std_path_smoothness.append(round(np.std(smoothness), 2))
        std_traveled_distance.append(round(np.std(path_distance), 2))
        print('--------------------------------------------------------------------')

    average_data = {
        'Test Case': ['Case 1', 'Case 2', 'Case 3', 'Case 4'],
        'Q-Learning_Time': average_time_taken_q,
        'IQFPA_Time': average_time_taken,
        '% of Improvement-Time': list(
            map(calculate_percentage_improvement, average_time_taken_q, average_time_taken)),
        'Q-Learning_distance': average_traveled_distance_q,
        'IQFPA_distance': average_traveled_distance,
        '% of Improvement-Distance': list(
            map(calculate_percentage_improvement, average_traveled_distance_q, average_traveled_distance)),
        'Q-learning_smoothness': average_path_smoothness_q,
        'IQFPA_smoothness': average_path_smoothness,
        '% of Improvement-Smoothness': list(
            map(calculate_percentage_improvement, average_path_smoothness_q, average_path_smoothness))
    }

    std_data = {
        'Test Case': ['Case 1', 'Case 2', 'Case 3', 'Case 4'],
        'Q-Learning_Time': std_time_taken_q,
        'IQFPA_Time': std_time_taken,
        '% of Improvement-Time': list(map(calculate_percentage_improvement, std_time_taken_q, std_time_taken)),
        'Q-Learning_distance': std_traveled_distance_q,
        'IQFPA-distance': std_traveled_distance,
        '% of Improvement-Distance': list(
            map(calculate_percentage_improvement, std_traveled_distance_q, std_traveled_distance)),
        'Q-Learning_Smoothness': std_path_smoothness_q,
        'IQFPA-smoothness': std_path_smoothness,
        '% of Improvement-Smoothness': list(
            map(calculate_percentage_improvement, std_path_smoothness_q, std_path_smoothness))
    }

    df1 = pd.DataFrame(average_data)
    df1.to_csv('results_average.csv', index=False)

    df2 = pd.DataFrame(std_data)
    df2.to_csv('results_std.csv', index=False)


if __name__ == '__main__':
    obstacles_8 = [(2, 6), (3, 5), (3, 6), (3, 7), (3, 8), (2, 7), (1, 6), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7),
                   (6, 7), (8, 11), (8, 13), (8, 12), (7, 12), (7, 17), (7, 18), (6, 18),
                   (6, 17), (7, 16), (5, 18), (17, 14), (18, 14), (16, 15), (15, 15),
                   (17, 15), (17, 16), (18, 16), (18, 17), (8, 0), (8, 1), (8, 2), (8, 3), (9, 2), (9, 1), (14, 3),
                   (14, 4), (14, 5), (15, 4), (15, 2), (15, 3), (17, 3), (17, 4), (17, 6), (17, 7), (17, 5), (18, 4),
                   (18, 5), (18, 6), (14, 9), (14, 10), (14, 11), (13, 10), (12, 10), (12, 9), (12, 8), (11, 8),
                   (11, 7)]
    # obstacles_12 = [(2, 6), (3, 5), (3, 6), (3, 7), (3, 8), (2, 7), (1, 6), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7),
    #                 (6, 7),
    #                 (8, 11), (8, 13), (8, 12), (7,12), (10, 11), (10, 12), (7, 17), (7, 18), (6, 18),
    #                 (6, 17), (7, 16), (5, 18), (17, 14), (18, 14), (16, 15), (15, 15),
    #                 (17, 15), (17, 16), (18, 16), (18, 17), (8, 0), (8, 1), (8, 2), (8, 3), (9, 2), (9, 1), (14, 3),
    #                 (14, 4), (14, 5), (15, 4), (15, 2), (15, 3), (17, 3), (17, 4), (17, 6), (17, 7), (17, 5), (18, 4),
    #                 (18, 5), (18, 6), (14, 9), (14, 10), (14, 11), (13, 10), (12, 10), (12, 9), (12, 8), (11, 8),
    #                 (11, 7),
    #                 (8, 6), (8, 7), (9, 6), (9, 7), (2,14), (3,14), (2,13), (10, 16), (11, 16)]
    obstacles_blocks = [(6, 6), (6, 9), (6, 12), (6, 15), (6, 18), (10, 4), (10, 7), (10, 10), (10, 13), (10, 16),
                          (15, 2), (15, 5), (15, 8), (15, 11), (15, 14)]

    obstacles_lines = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
                       (11, 11),
                       (12, 12), (14, 4), (15, 5), (16, 6), (17, 7), (18, 8), (19, 9), (4, 14), (5, 15), (6, 16),
                       (7, 17), (8, 18), (9, 19)]

    obstacles_lines2 = [(0,14), (1, 14), (2, 14), (3, 14), (7, 6), (7,7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
                        (7, 10), (8, 10), (9, 10), (11, 10), (12, 10), (13, 10), (14, 10), (10, 0), (10, 1), (10, 2),
                        (10, 3), (10, 4), (10, 5), (10, 6), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19),
                        (14, 4), (14, 5), (14, 6), (15, 4), (16, 4), (17, 4), (18, 4), (19, 4)]

    obstacle_list = [obstacles_8, obstacles_blocks, obstacles_lines, obstacles_lines2]
    n_flowers = int(input('Enter population size for FPA: '))
    initialize_and_process(obstacle_list, n_flowers)
