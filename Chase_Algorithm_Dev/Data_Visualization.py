import numpy as np
import matplotlib.pyplot as plt


def plot_xy(t_list, pose_list):
    t = t_list
    pose = pose_list

    x = pose[:, 0]
    y = pose[:, 1]

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # 1) X vs Y (trajectory)
    axs[0].plot(x, y)
    axs[0].set_title("X vs Y Trajectory")
    axs[0].set_xlabel("X [mm]")
    axs[0].set_ylabel("Y [mm]")
    axs[0].axis("equal")
    axs[0].grid(True)

    # 2) X vs Time
    axs[1].plot(t, x)
    axs[1].set_title("X vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("X [mm]")
    axs[1].grid(True)

    # 3) Y vs Time
    axs[2].plot(t, y)
    axs[2].set_title("Y vs Time")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Y [mm]")
    axs[2].grid(True)

    plt.tight_layout()
#     plt.show()

def plot_velocities(t, lin_vel, ang_vel):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Linear velocity plot
    axs[0].plot(t, lin_vel)
    axs[0].set_title("Linear Velocity vs Time")
    axs[0].set_ylabel("Linear Velocity [mm/s]")
    axs[0].grid(True)

    # Angular velocity plot
    axs[1].plot(t, ang_vel)
    axs[1].set_title("Angular Velocity vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular Velocity [rad/s]")
    axs[1].grid(True)

    plt.tight_layout()
#     plt.show()

def plot_accelerations(t, lin_acc, ang_acc):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Linear acceleration plot
    axs[0].plot(t, lin_acc)
    axs[0].set_title("Linear Acceleration vs Time")
    axs[0].set_ylabel("Linear Acceleration [mm/s²]")
    axs[0].grid(True)

    # Angular acceleration plot
    axs[1].plot(t, ang_acc)
    axs[1].set_title("Angular Acceleration vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular Acceleration [rad/s²]")
    axs[1].grid(True)

    plt.tight_layout()
#     plt.show()
