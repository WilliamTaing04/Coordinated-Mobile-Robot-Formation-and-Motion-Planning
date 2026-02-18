import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Utility: Rolling Average
# -------------------------------
def rolling_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')


# -------------------------------
# 1) XY Trajectory Plot
# -------------------------------
def plot_xy_trajectory(pose_list):
    pose = pose_list
    x = pose[:, 0]
    y = pose[:, 1]

    plt.figure(figsize=(6,6))
    plt.plot(x, y)
    plt.title("X vs Y Trajectory")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()


# -------------------------------
# 2) X vs T and Y vs T (2 subplots)
# -------------------------------
def plot_xy_vs_time(t_list, pose_list):
    t = t_list
    pose = pose_list
    x = pose[:, 0]
    y = pose[:, 1]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(t, x)
    axs[0].set_title("X vs Time")
    axs[0].set_ylabel("X [mm]")
    axs[0].grid(True)

    axs[1].plot(t, y)
    axs[1].set_title("Y vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Y [mm]")
    axs[1].grid(True)

    plt.tight_layout()


# -------------------------------
# 3) Velocity Plots (with goals + smoothing)
# -------------------------------
def plot_velocities(t, lin_vel, ang_vel,
                    v_goal=None, w_goal=None,
                    window=20):

    lin_smooth = rolling_average(lin_vel, window)
    ang_smooth = rolling_average(ang_vel, window)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Linear velocity
    axs[0].plot(t, lin_vel, alpha=0.4, label="Raw")
    axs[0].plot(t, lin_smooth, linewidth=2, label="Rolling Avg")

    if v_goal is not None:
        axs[0].axhline(v_goal, linestyle='--', linewidth=2, label="V Goal")

    axs[0].set_title("Linear Velocity vs Time")
    axs[0].set_ylabel("Linear Velocity [mm/s]")
    axs[0].grid(True)
    axs[0].legend()

    # Angular velocity
    axs[1].plot(t, ang_vel, alpha=0.4, label="Raw")
    axs[1].plot(t, ang_smooth, linewidth=2, label="Rolling Avg")

    if w_goal is not None:
        axs[1].axhline(w_goal, linestyle='--', linewidth=2, label="W Goal")

    axs[1].set_title("Angular Velocity vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular Velocity [rad/s]")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()


# -------------------------------
# 4) Acceleration Plots (with smoothing)
# -------------------------------
def plot_accelerations(t, lin_acc, ang_acc, window=25):

    lin_smooth = rolling_average(lin_acc, window)
    ang_smooth = rolling_average(ang_acc, window)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(t, lin_acc, alpha=0.4, label="Raw")
    axs[0].plot(t, lin_smooth, linewidth=2, label="Rolling Avg")
    axs[0].set_title("Linear Acceleration vs Time")
    axs[0].set_ylabel("Linear Acceleration [mm/s²]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, ang_acc, alpha=0.4, label="Raw")
    axs[1].plot(t, ang_smooth, linewidth=2, label="Rolling Avg")
    axs[1].set_title("Angular Acceleration vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular Acceleration [rad/s²]")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
