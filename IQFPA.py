import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        q = current_q + lr * (reward + (gamma * np.max(self.q_table[new_r, new_col])) - current_q)
        return q

    def q_learning(self):
        epsilon = 0.8
        total_reward = 0
        steps = 0
        reached_goal = False
        max_steps = 1000
        for i in range(self.max_iter):
            row, column = self.start[0], self.start[1]
            steps = 0
            while (row, column) != self.goal and steps < max_steps:
                a = self.get_action(row, column, epsilon)
                new_r, new_col = self.get_next_state(row, column, a)
                reward = self.rewards[new_r, new_col]
                total_reward += reward
                lr = 0.2
                gamma = 0.8
                current_q = self.q_table[row, column, a]
                q = current_q + lr * (reward + (gamma * np.max(self.q_table[new_r, new_col])) - current_q)
                self.q_table[row, column, a] = q

                row = new_r
                column = new_col
                steps += 1

                if (row, column) == self.goal:
                    reached_goal = True
                    break

                # if steps == max_steps:
                #     print(f"Episode {i + 1} reached max steps limit. Consider checking state transitions or rewards.")

        return round(total_reward, 2), steps, reached_goal

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
        plt.savefig(f'{name}_{iteration}.png')
        plt.close()

    def get_policy(self):
        policy_grid = np.full((self.n_row, self.n_column), -1)
        row, column = self.start[0], self.start[1]
        policy = [(row, column)]

        while (row, column) != self.goal:
            a = np.argmax(self.q_table[row, column])
            policy_grid[row, column] = a
            row, column = self.get_next_state(row, column, a)
            policy.append((row, column))

            if (row, column) in self.obstacles:
                break

        policy_grid[self.goal[0], self.goal[1]] = 6
        for x in self.obstacles:
            policy_grid[x[0], x[1]] = 5

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


def initialize_and_process(obstacles_list, population_size, n_iterations):
    q_learning_max_iter = 150

    steps = []
    rewards = []
    reached = []
    train_time = []
    steps_1 = []
    rewards_1 = []
    reached_1 = []
    train_time_1 = []

    start = (2, 16)
    goal = (19, 11)

    for idx, obstacles in enumerate(obstacles_list):
        print(f'Test case {idx}')
        total_time_q = []
        path_distance_q = []
        smoothness_q = []

        for i in range(n_iterations):
            print(f'Q-learning iteration {i}')
            grid = np.zeros((20, 20))
            q_learning = FPA(20, 20, 4, grid, start, goal, obstacles, population_size, q_learning_max_iter)

            # Calculating time - Q learning
            start_time = time.time()
            total_reward, step, reached_goal = q_learning.q_learning()
            t = round(time.time() - start_time, 2)
            train_time.append(t)
            steps.append(step)
            rewards.append(total_reward)
            reached.append(reached_goal)
            total_time_q.append(t)

            # if reached_goal:
            #     policy = q_learning.get_policy()
            #     path_distance_q.append(calculate_total_traveled_distance(policy))
            #     smoothness_q.append(calculate_path_smoothness(policy))
            #     q_learning.plot_grid_world(f'Q-learning_{idx + 1}', idx, i, policy)

        total_time = []
        path_distance = []
        smoothness = []
        for i in range(n_iterations):
            print(f'IQFPA iteration {i}')
            grid = np.zeros((20, 20))
            fpa = FPA(20, 20, 4, grid, start, goal, obstacles, population_size, q_learning_max_iter)

            # Calculating time - FPA + Q learning
            start_time = time.time()
            fpa.fpa_algorithm()
            total_reward, step, reached_goal = fpa.q_learning()
            t = round(time.time() - start_time, 2)
            train_time_1.append(t)
            steps_1.append(step)
            rewards_1.append(total_reward)
            reached_1.append(reached_goal)
            total_time.append(t)

            if reached_goal:
                policy = fpa.get_policy()
                path_distance.append(calculate_total_traveled_distance(policy))
                smoothness.append(calculate_path_smoothness(policy))
                fpa.plot_grid_world(f'IQFPA_{idx + 1}', idx, i, policy)

        print('--------------------------------------------------------------------')

    data = {
        "Steps": steps,
        "Rewards": rewards,
        "Reached Goal": reached,
        "Time to train": train_time,
        "Steps_IQFPA": steps_1,
        "Rewards_IQFPA": rewards_1,
        "Reached Goal_IQFPA": reached_1,
        "Time to train_IQFPA": train_time_1,
    }

    df = pd.DataFrame(data)
    df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    # obstacles = [(0, 6), (1, 6), (2, 3), (2, 6), (3, 3), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 3), (4, 11), (5, 3), (5, 11), (6, 3), (6, 11), (7, 3), (7, 11), (8, 3), (8, 11), (9, 3), (9, 11), (9, 12), (9, 13), (10, 3), (10, 13), (11, 3), (11, 13), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 13), (13, 9), (13, 13), (14, 9), (14, 13), (15, 9), (15, 13), (16, 9), (16, 13), (17, 9), (17, 13), (18, 9), (19, 9)]
    # obstacles = [(0, 1), (1, 2), (2, 3), (2, 10), (3, 3), (3, 10), (4, 3), (5, 2), (6, 2), (7, 2), (7, 7), (7, 8), (8, 2), (8, 7), (9, 4), (9, 7), (10, 4), (10, 6), (11, 4), (12, 4), (12, 12), (12, 13), (12, 14), (12, 15), (13, 4), (13, 13), (14, 13), (15, 13), (16, 5), (16, 13), (17, 5), (18, 5), (19, 5)]
    # obstacles_1 = [(4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 7), (5, 11), (6, 7), (6, 11), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (8, 6), (8, 12), (9, 6), (9, 12), (10, 6), (10, 12), (11, 6), (11, 12), (12, 6), (12, 12), (13, 6), (13, 12), (14, 6), (14, 12), (15, 6), (15, 7), (15, 8), (15, 9), (15, 10), (15, 11), (15, 12)]
    # obstacles = []
    obstacles = [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (4, 10), (5, 8), (5, 10), (6, 8), (6, 10), (7, 10), (8, 10), (9, 8), (9, 10), (10, 8), (10, 10), (11, 8), (11, 10), (12, 8), (13, 8), (14, 8), (15, 8), (16, 8), (17, 8), (18, 8), (19, 8)]
    obstacle_list = [obstacles]
    n_flowers = 50 # Paper suggests using size 10-50. 50 provides best results.
    n_iterations = int(input("Enter number of times to run each test case: "))
    initialize_and_process(obstacle_list, n_flowers, n_iterations)
