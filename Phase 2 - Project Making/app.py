import tkinter as tk
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the maze
ROWS = 6
COLS = 6
OBSTACLES = [(1, 1), (2, 2), (3, 1), (4, 3)]
GOAL = (5, 5)

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
EPISODES = 300

class MazeGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Q-Learning Maze Game")
        self.canvas = tk.Canvas(master, width=600, height=600)
        self.canvas.pack()
        self.create_grid()
        self.reset()

        self.q_table = np.zeros((ROWS, COLS, 4)) 
        # 4 actions: up, down, left, right
        self.actions = ['up', 'down', 'left', 'right']
        
        self.training = False
        
        # Store the total reward per episode for the graph
        self.episode_rewards = []
        
        # Label to show current episode number
        self.episode_label = tk.Label(master, text="Episode: 0")
        self.episode_label.pack()
        
        # Create the graph to show learning improvement
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Learning Progress (Total Reward per Episode)')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_graph.get_tk_widget().pack()
    
    def create_grid(self):
        # Draw grid
        self.cells = {}
        for i in range(ROWS):
            for j in range(COLS):
                x1, y1 = j * 100, i * 100
                x2, y2 = x1 + 100, y1 + 100
                self.cells[(i, j)] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
        
        # Draw obstacles
        for obs in OBSTACLES:
            self.canvas.itemconfig(self.cells[obs], fill="black")

        # Draw goal
        self.canvas.itemconfig(self.cells[GOAL], fill="green")

    def reset(self):
        self.agent_pos = (0, 0)
        self.update_agent()

    def update_agent(self):
        for cell in self.cells:
            if cell == self.agent_pos:
                self.canvas.itemconfig(self.cells[cell], fill="blue")
            elif cell == GOAL:
                self.canvas.itemconfig(self.cells[cell], fill="green")
            elif cell in OBSTACLES:
                self.canvas.itemconfig(self.cells[cell], fill="black")
            else:
                self.canvas.itemconfig(self.cells[cell], fill="white")

    def step(self, action):
        i, j = self.agent_pos
        if action == 'up' and i > 0: 
            i -= 1
        elif action == 'down' and i < ROWS - 1:
            i += 1
        elif action == 'left' and j > 0: 
            j -= 1
        elif action == 'right' and j < COLS - 1:
            j += 1
        next_pos = (i, j)
        reward = -1
        done = False

        if next_pos in OBSTACLES:
            next_pos = self.agent_pos  # Stay in place
        elif next_pos == GOAL:
            reward = 10
            done = True

        self.agent_pos = next_pos
        self.update_agent()
        return next_pos, reward, done

    def train_agent(self):
        self.training = True
        for episode in range(EPISODES):
            self.reset()
            state = self.agent_pos
            done = False
            total_reward = 0

            while not done:
                if random.uniform(0, 1) < EPSILON:
                    action = random.choice(self.actions)
                else:
                    action = self.actions[np.argmax(self.q_table[state[0], state[1]])]

                next_state, reward, done = self.step(action)
                action_idx = self.actions.index(action)

                # Q-learning update
                best_next_action = np.max(self.q_table[next_state[0], next_state[1]])
                self.q_table[state[0], state[1], action_idx] += ALPHA * (
                    reward + GAMMA * best_next_action - self.q_table[state[0], state[1], action_idx]
                )
                state = next_state
                total_reward += reward

                if self.training:
                    time.sleep(0.1)
                    self.master.update()

            # Append the total reward of this episode
            self.episode_rewards.append(total_reward)

            # Update episode label
            self.episode_label.config(text=f"Episode: {episode+1}")

            # Update graph
            self.ax.plot(range(1, episode+2), self.episode_rewards, color='blue')
            self.canvas_graph.draw()

        self.training = False
        self.master.update()

    def start_training(self):
        self.train_agent()
        print("Training Complete!")

# Run the game
def main():
    root = tk.Tk()
    game = MazeGame(root)

    tk.Button(root, text="Start Training", command=game.start_training).pack()
    root.mainloop()

if __name__ == "__main__":
    main()
