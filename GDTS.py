import math
import numpy as np
import matplotlib.pyplot as plt 
import time 

class GDTS:
    def __init__(self, turn_rate, turn_radius = 10, strength=10000):
        self.turn_radius = turn_radius
        self.airspeed = turn_rate * self.turn_radius
        self.turn_rate = np.degrees(turn_rate)
        self.strength = strength
        self.trans_loc = [150, 200]
        self.total_path_coordinates_x = []
        self.total_path_coordinates_y = []
        self.initialization()

    def signal_strength(self, pos):
        x, y = pos
        var_x, var_y = 170**2, 170**2
        exponent = -(((x - self.trans_loc[0])**2 / (2 * var_x)) + ((y - self.trans_loc[1])**2 / (2 * var_y)))
        curr_strength = self.strength * math.exp(exponent)
        return curr_strength

    def initialization(self):
        self.strengths = []
        self.loop = 1
        self.timestep = 0
        self.del_t = 1/self.turn_radius
        self.threshold_dist = self.turn_radius    # self.turn_radius
        self.uav_pos = []

        # Initial receiver location
        x, y = map(float, input("Enter initial X, Y coordinates separated by space of Receiver: ").split())
        self.initial_rec_loc = [x, y]
        self.curr_pos = [x, y]
        self.grad_dir = []
        self.x_vel = 0
        self.y_vel = 0
        self.p = [0, 0]
        self.w = self.turn_rate
        self.uav_pos.append(self.curr_pos)
        self.heading = 0
        self.delta = 2

        self.iter_loop()

    def gradient_direction(self, uav_pos, strengths):
        uav_pos = np.array(uav_pos)
        A = np.hstack((uav_pos, np.ones((len(strengths), 1))))
        B = np.array(strengths).reshape(-1, 1)
        gradient = np.linalg.lstsq(A, B, rcond=None)[0]
        print(f"Gradient: {gradient[:2].flatten()}")
        return gradient[:2].flatten() 
    

    def iter_loop(self):
        curr_dist = np.sqrt((self.trans_loc[0] - self.uav_pos[-1][0])**2 + (self.trans_loc[1] - self.uav_pos[-1][1])**2)
        print("Initial Distance: ", curr_dist)

        self.strengths.append(self.signal_strength(self.curr_pos))

        x, y = self.curr_pos
        self.total_path_coordinates_x.append(x)
        self.total_path_coordinates_y.append(y)

        count = 1

        while curr_dist > self.threshold_dist:
            self.timestep += 1

            if count >= 45000 :
                break 

            count += 1

            # Update heading
            self.heading = self.heading + (self.turn_rate * self.del_t)
            self.heading = (self.heading + 180) % 360 - 180

            self.x_vel = self.airspeed * np.cos(np.radians(self.heading))
            self.y_vel = self.airspeed * np.sin(np.radians(self.heading))

            x += self.x_vel * self.del_t
            y += self.y_vel * self.del_t

            self.total_path_coordinates_x.append(x)
            self.total_path_coordinates_y.append(y)

            self.strengths.append(self.signal_strength([x, y]))
            self.uav_pos.append([x, y])

            self.p = np.array([x, y]) - np.array(self.curr_pos)

            print(f"Timestep: {self.timestep}, Heading: {self.heading}, Turn Rate: {self.turn_rate}, Current Position: ({x}, {y})")

            if self.loop == 1:
                if self.timestep > 2 and (self.strengths[self.timestep - 2] < self.strengths[self.timestep - 1] and self.strengths[self.timestep - 1] > self.strengths[self.timestep]):
                    self.turn_rate = -self.turn_rate

                    self.grad_dir = self.gradient_direction(self.uav_pos, self.strengths)
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []
                    self.timestep = 0
                    self.loop += 1
                    print("Turn Rate Changed After First Loop")

            else:
                angle_diff = np.degrees(np.arccos(np.dot(self.p, self.grad_dir) / (np.linalg.norm(self.p) * np.linalg.norm(self.grad_dir))))
                if abs(angle_diff) < self.delta:
                    self.grad_dir = self.gradient_direction(self.uav_pos, self.strengths)
                    theta = np.degrees(np.arctan2(self.grad_dir[1], self.grad_dir[0]))

                    if np.sign(self.heading - theta) == np.sign(self.turn_rate):
                        self.turn_rate = -self.turn_rate
                        print(f"Turn rate Changed After {self.loop} loop") 

                    if np.linalg.norm(np.array(self.uav_pos[-1]) - np.array(self.curr_pos)) <= 0.9:
                        self.turn_rate = -self.turn_rate
                        self.grad_dir = self.gradient_direction(self.uav_pos, self.strengths)
                        print("Turn rate Changed due to Complete Circular Trajectory")

                    self.loop += 1
                    self.timestep = 0
                    self.strengths = []
                    self.curr_pos = self.uav_pos[-1]
                    self.uav_pos = []

            curr_dist = np.sqrt((self.trans_loc[0] - x)**2 + (self.trans_loc[1] - y)**2)

        self.plotting_path()

    def plotting_path(self):

        

        plt.figure(figsize=(10, 10))
        plt.scatter(self.total_path_coordinates_x, self.total_path_coordinates_y, c='r', s=2, label='Path')
        plt.scatter([self.trans_loc[0]], [self.trans_loc[1]], c='b', s=50, label='Transmitter')
        plt.scatter([self.initial_rec_loc[0]], [self.initial_rec_loc[1]], c='g', s=50, label='Receiver')
        plt.annotate("Rec", xy=(self.initial_rec_loc[0] - 2, self.initial_rec_loc[1] - 2))
        plt.annotate("Tran", xy=(self.trans_loc[0], self.trans_loc[1]))
        plt.annotate(f"Turn Radius: {self.turn_radius}", xy=(self.trans_loc[0] + 20, self.trans_loc[1] + 10))
        plt.xlabel('X - coordinates')
        plt.ylabel('Y - coordinates')
        plt.title('Total Path Travelled Using GDTS')
        plt.legend()
        plt.savefig("GDTS_sim_05_3.png")
        plt.show()

if __name__ == '__main__':
    turn_rate = float(input('Enter the Turn Rate of Receiver (rad/sec): '))
    GDTS(turn_rate=turn_rate)