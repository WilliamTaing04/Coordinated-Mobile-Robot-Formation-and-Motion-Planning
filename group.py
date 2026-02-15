import numpy as np
from math import *
import matplotlib.pyplot as plt
from agent import Agent
from controller import (
    SafeFormationController,
    SinosodialController,
    ConstantController,
    TrajectoryController,
)

from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.io import savemat

plt.rc('legend', frameon=False)
COLOR_LIST = ["r", "g", "b", "k", "c", "p"]
ALIAS_LIST = ["Leader", "Follower-1", "Follower-2", "Follower-3"]
MAPX = [-2, 4.5]
MAPY = [-3.0, 3.0]

EW = 1.4
EU = 1.4
DR = 0.3
T = 1.0
ALPHA = 5.0
INF = 100

WHEEL_SPEED_RPM = 200
WHEEL_RADIUS = 0.05
BOT_RADIUS = 0.08
LINEAR_SPEED = WHEEL_SPEED_RPM * 2 * pi * WHEEL_RADIUS / 60
ANGULAR_SPEED = LINEAR_SPEED / BOT_RADIUS

V_MAX = LINEAR_SPEED  # 0.5
U_MAX = 0.5
W_MAX = 2.0  # ANGULAR_SPEED  # 1


def animate(frame, group=None, axes=None):
    axes.clear()
    axes.set_xlim(MAPX)
    axes.set_ylim(MAPY)
    for id in range(group.n_agents):
        x, y = group.agent_list[id].get_history_pose(frame * 10)
        axes.plot(x, y, COLOR_LIST[id], marker="o")


class Group:
    def __init__(self, n_agents=1, safety=False, state=None, controller=None):
        self.agent_list = []
        self.n_agents = n_agents
        for idx in range(n_agents):
            state_i = None
            controller_i = None
            if state is not None:
                state_i = state[idx]
            if controller is not None:
                controller_i = controller[idx]
            self.agent_list.append(
                Agent(
                    _id=idx,
                    _cluster_size=n_agents,
                    safety=safety,
                    state=state_i,
                    controller=controller_i,
                )
            )
        self.group_pos = np.zeros((n_agents, 2))
        self.group_pos_history = []
        self.folder = "./tmp_images"

    def test(self):
        for idx in range(self.n_agents):
            self.group_pos[idx, :] = self.agent_list[idx].get_pos()
        # print(self.group_pos)
        for idx, agent in enumerate(self.agent_list):
            agent.init_estimates(self.group_pos)
            # print(agent.get_position_estimates())

    def run_experiment(self, t_end=10.0, t_jump=0.001, visualize=False, save=False):
        t = 0
        for idx in range(self.n_agents):
            self.group_pos[idx, :] = self.agent_list[idx].get_pos()

        for idx, agent in enumerate(self.agent_list):
            agent.init_estimates(self.group_pos)
            # print(agent.get_position_estimates())
        if visualize:
            fig, ax = plt.subplots(1, self.n_agents + 1, figsize=(24, 6), dpi=80)
            for ax_id in ax:
                ax_id.grid()
            ax[-1].set_xlim([-2, 2])
            ax[-1].set_ylim([-2, 2])
        new_pos = np.copy(self.group_pos)
        image_idx = 0
        while t < t_end:
            for idx, agent in enumerate(self.agent_list):
                agent.set_observations(self.group_pos)
                agent.RK4_step(t_jump)
                new_pos[idx, :] = agent.get_pos()
                if visualize:
                    agent.visualize(ax)
                # print(agent.get_position_estimates())
            t += t_jump
            self.group_pos_history.append(new_pos)
            self.group_pos = np.copy(new_pos)
            if visualize:
                plt.show(block=False)
                plt.pause(t_jump / 10)
                for ax_id in ax:
                    ax_id.clear()
                    ax_id.grid()
                ax[-1].set_xlim([-2.5, 2.5])
                ax[-1].set_ylim([-2.5, 2.5])
            if save:
                fig.savefig(self.folder + "/file%03d.png" % image_idx)
                image_idx += 1

    def plot_history(self):
        fig, ax = plt.subplots(self.n_agents, 6)
        col = [
            "Motion",
            "Velocity",
            "Control-u",
            "Control-w",
            "Safety Along motion",
            "Safety Perpendicular to motion",
        ]
        for id, axess in enumerate(ax[0]):
            axess.set_title(col[id])
        for i in range(self.n_agents):
            self.agent_list[i].plot_history(ax)
        for ax_id in ax:
            ax_id[0].grid()
            ax_id[1].grid()
            ax_id[2].grid()
            ax_id[3].grid()
            ax_id[4].grid()
            ax_id[5].grid()
        # fig.savefig("Experiment3.png", format="png", dpi=1000)
        plt.show()

    def run_animation(self, frames, folder_name="/"):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        axes.set_xlim([-2, 4])
        axes.set_ylim([-2, 4])
        anim = FuncAnimation(
            fig, animate, fargs=(self, axes), interval=1, frames=int(frames)
        )
        f = folder_name + "/animation.gif"
        writergif = animation.PillowWriter(fps=30)
        anim.save(f, writer=writergif)
        return

    def save_motion(self, folder_name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        axes.set_xlim(MAPX)
        axes.set_ylim(MAPY)
        file_name = folder_name + "Motion"
        data_dict = {}
        for id in range(self.n_agents):
            pos = self.agent_list[id].experiment_history[:, :, id]
            data_dict["x"+str(id)] = pos[:, 0]
            data_dict["y"+str(id)] = pos[:,1]
            axes.plot(pos[:, 0], pos[:, 1], COLOR_LIST[id], linewidth=2)
        axes.grid()
        axes.tick_params(axis="y", which="both", direction="in", right=True)
        axes.set_title("Motion of the Formation")
        axes.legend(ALIAS_LIST)
        axes.set_xlabel("Position X (m)")
        axes.set_ylabel("Position Y (m)")
        savemat(file_name + ".mat",data_dict)
        fig.savefig(file_name+".png", format="png", dpi=1000)
        plt.show()

    def save_velocity(self, t_end=0, folder_name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        axes.set_xlim([0, t_end])
        axes.set_ylim([0, 0.1 + V_MAX])
        file_name = folder_name + "Velocity"
        data_dict = {}
        for id in range(self.n_agents):
            pos = self.agent_list[id].experiment_history[:, :, id]
            data_dict["x"+str(id)] = self.agent_list[id].history_stamp
            data_dict["y"+str(id)] = pos[:,2]
            axes.plot(
                self.agent_list[id].history_stamp,
                pos[:, 2],
                COLOR_LIST[id],
                linewidth=2,
            )
        axes.grid()
        axes.tick_params(axis="y", which="both", direction="in", right=True)
        axes.set_title("Velocity of individual Agent")
        axes.legend(ALIAS_LIST)
        axes.set_xlabel("time (sec)")
        axes.set_ylabel("velocity (m/sec)")
        savemat(file_name + ".mat",data_dict)
        fig.savefig(file_name+".png", format="png", dpi=1000)
        plt.show()

    def save_control_and_safety(self, t_end=0, folder_name=""):
        if True:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            axes.set_xlim([0, t_end])
            axes.set_ylim([-U_MAX - 0.1, U_MAX + 0.1])
            file_name = folder_name + "Control1"
            data_dict = {}
            for id in range(self.n_agents):
                pos = self.agent_list[id].control_history
                data_dict["x"+str(id)] = self.agent_list[id].history_stamp
                data_dict["y"+str(id)] = pos[:,0]
                axes.plot(
                    self.agent_list[id].history_stamp,
                    pos[:, 0],
                    COLOR_LIST[id],
                    linewidth=2,
                )
            axes.grid()
            axes.tick_params(axis="y", which="both", direction="in", right=True)
            axes.set_title("Acceleration of individual Agent")
            axes.legend(ALIAS_LIST)
            axes.set_xlabel("time (sec)")
            axes.set_ylabel("Acceleration (m/sec^2)")
            savemat(file_name + ".mat",data_dict)
            fig.savefig(file_name+".png", format="png", dpi=1000)
            plt.show()
        if True:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            axes.set_xlim([0, t_end])
            axes.set_ylim([-W_MAX - 0.1, W_MAX + 0.1])
            file_name = folder_name + "Control2"
            data_dict = {}
            for id in range(self.n_agents):
                pos = self.agent_list[id].control_history
                data_dict["x"+str(id)] = self.agent_list[id].history_stamp
                data_dict["y"+str(id)] = pos[:,1]
                axes.plot(
                    self.agent_list[id].history_stamp,
                    pos[:, 1],
                    COLOR_LIST[id],
                    linewidth=2,
                )
            axes.grid()
            axes.tick_params(axis="y", which="both", direction="in", right=True)
            axes.set_title("Angular Velocity of individual Agent")
            axes.legend(ALIAS_LIST)
            axes.set_xlabel("time (sec)")
            axes.set_ylabel("Angular Velocity (rad/sec)")
            savemat(file_name+".mat",data_dict)
            fig.savefig(file_name+".png", format="png", dpi=1000)
            plt.show()
        if True:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            axes.set_xlim([0, t_end])
            axes.set_ylim([0, 0.6])
            file_name = folder_name + "Safety1"
            data_dict = {}
            for id in range(1, self.n_agents):
                pos = self.agent_list[id].control_history
                data_dict["x"+str(id)] = self.agent_list[id].history_stamp
                data_dict["y"+str(id)] = pos[:,2]
                axes.plot(
                    self.agent_list[id].history_stamp,
                    pos[:, 2],
                    COLOR_LIST[id],
                    linewidth=2,
                )
            axes.grid()
            axes.tick_params(axis="y", which="both", direction="in", right=True)
            axes.set_title("Safety along motion for individual Agent")
            axes.legend(ALIAS_LIST[1:])
            axes.set_xlabel("time (sec)")
            axes.set_ylabel("Safety along motion (m)")
            savemat(file_name+".mat",data_dict)
            fig.savefig(file_name+".png", format="png", dpi=1000)
            plt.show()
        if True:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            axes.set_xlim([0, t_end])
            axes.set_ylim([0, 0.25])
            file_name = folder_name + "Safety2"
            for id in range(1, self.n_agents):
                pos = self.agent_list[id].control_history
                data_dict["x"+str(id)] = self.agent_list[id].history_stamp
                data_dict["y"+str(id)] = pos[:,3]
                axes.plot(
                    self.agent_list[id].history_stamp,
                    pos[:, 3],
                    COLOR_LIST[id],
                    linewidth=2,
                )
            axes.grid()
            axes.tick_params(axis="y", which="both", direction="in", right=True)
            axes.set_title("Safety perpendicular to motion for individual Agent")
            axes.legend(ALIAS_LIST[1:])
            axes.set_xlabel("time (sec)")
            axes.set_ylabel("Safety perpendicular to motion (m)")
            savemat(file_name+".mat",data_dict)
            fig.savefig(file_name+".png", format="png", dpi=1000)
            plt.show()

    def save_estimates(self, t_end, folder_name=""):
        fig, axes = plt.subplots(nrows=self.n_agents, ncols=2, figsize=(12, 16))
        for ax in axes:
            ax[0].set_xlim([0, t_end])
            ax[1].set_xlim([0, t_end])
            ax[0].set_ylim([-0.001, 0.025])
            ax[1].set_ylim([-0.1, 0.25])
        file_name = folder_name + "Estimate"
        data_dict = {}
        for id in range(self.n_agents):
            pos = self.agent_list[id].experiment_history[:, :, id]
            alias = []
            for id_1 in range(self.n_agents):
                if id_1 == id:
                    continue
                pos1 = self.agent_list[id_1].experiment_history[:, :, id]
                pos_val = pos1[:, :2] - pos[:, :2]
                vel_mag = pos1[:, 2] - pos[:, 2]
                pos_mag = np.linalg.norm(pos_val, axis=1)
                data_dict["x"+str(id)+str(id_1)] = self.agent_list[id_1].history_stamp
                data_dict["pos"+str(id)+str(id_1)] = pos_mag
                data_dict["vel"+str(id)+str(id_1)] = vel_mag
                axes[id][0].plot(
                    self.agent_list[id_1].history_stamp,
                    pos_mag,
                    COLOR_LIST[id_1] + "--",
                    linewidth=2,
                )
                axes[id][1].plot(
                    self.agent_list[id_1].history_stamp,
                    vel_mag,
                    COLOR_LIST[id_1] + "--",
                    linewidth=2,
                )
                alias.append(ALIAS_LIST[id_1])
            axes[id][0].legend(alias)
            axes[id][1].legend(alias)
            axes[id][0].grid()
            axes[id][1].grid()
            axes[id][0].tick_params(axis="y", which="both", direction="in", right=True)
            axes[id][1].tick_params(axis="y", which="both", direction="in", right=True)
            axes[id][0].set_xlabel("Time (sec)")
            axes[id][0].set_ylabel("Error for " + ALIAS_LIST[id])
            axes[id][1].set_xlabel("Time (sec)")
            axes[id][1].set_ylabel("Error for " + ALIAS_LIST[id])
        axes[0][0].set_title("Position Estimation Error - Case 3")
        axes[0][1].set_title("Velocity Estimation Error - Case 3")
        fig.savefig(file_name+".png", format="png", dpi=1000)
        plt.show()
        savemat(file_name+".mat",data_dict)


def read_test(idx):
    t_end = 10
    agents = 3
    file_name = None
    if idx == 0:
        file_name = "string_stability.txt"
    if idx == 1:
        file_name = "trajectory.txt"
    if idx == 2:
        file_name = "circular.txt"
    if idx == 3:
        file_name = "estimator.txt"

    if file_name is None:
        return (
            agents,
            [[1, 1, 0, 3.14], [-1, -1, 0, 1.5], [1, -1, 0, 3.14]],
            None,
            t_end,
        )

    f = open(file_name, "r")
    lines = f.readlines()
    state = None
    controller = None
    for line in lines:
        lst = line.split(" ")
        if lst[0] == "\n":
            break
        elif lst[0] == "S":
            float_list = [float(x) for x in lst[1:]]
            id = int(float_list[0])
            state[id] = float_list[1:]

        elif lst[0] == "C":
            float_list = [float(x) for x in lst[1:]]
            id = int(float_list[0])
            cntrl_type = int(float_list[1])
            if cntrl_type == 0:
                controller[id] = SafeFormationController(float_list[2:])
            if cntrl_type == 1:
                controller[id] = TrajectoryController(float_list[2:])
            elif cntrl_type == 2:
                controller[id] = SinosodialController(float_list[2:])
            elif cntrl_type == 3:
                controller[id] = ConstantController(float_list[2:])
        else:
            agents = int(lst[0])
            t_end = float(lst[1])
            state = [[0, 0, 0, 0]] * agents
            controller = [[None]] * agents

    folder_name = file_name.split(".")[0] + "/"

    return agents, state, controller, t_end, folder_name


if __name__ == "__main__":
    agents, state, controller, t_end, folder_name = read_test(2)
    #folder_name = "new_" + folder_name
    g = Group(agents, safety=True, state=state, controller=controller)
    g.run_experiment(t_end=t_end, visualize=False, save=False)
    g.plot_history()
    # g.run_animation(frames=t_end * 10, folder_name=folder_name)
    g.save_motion(folder_name)
    g.save_velocity(t_end=t_end, folder_name=folder_name)
    g.save_estimates(t_end=t_end, folder_name=folder_name)
    g.save_control_and_safety(t_end=t_end, folder_name=folder_name)
    # os.chdir("./tmp_images")
    # subprocess.call(
    #     [
    #         "ffmpeg",
    #         "-framerate",
    #         "8",
    #         "-i",
    #         "file%02d.png",
    #         "-r",
    #         "30",
    #         "-pix_fmt",
    #         "yuv420p",
    #         "video_name.mp4",
    #     ]
    # )
