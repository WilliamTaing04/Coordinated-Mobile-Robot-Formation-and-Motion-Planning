import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Utility: Rolling Average
# -------------------------------
def rolling_average(x, window_size):
    if window_size <= 1:
        return np.array(x)
    # pad to keep same length behavior similar to 'same' conv
    w = int(window_size)
    return np.convolve(x, np.ones(w)/w, mode='same')

# -------------------------------
# 1) XY Trajectory Plot
# -------------------------------
def plot_xy_trajectory(pose_list):
    pose = np.asarray(pose_list)
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
    # plt.show()

# -------------------------------
# 2) X vs T and Y vs T (2 subplots)
# -------------------------------
def plot_xy_vs_time(t_list, pose_list):
    t = np.asarray(t_list)
    pose = np.asarray(pose_list)
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
    # plt.show()

# -------------------------------
# 3) Velocity Plots (with goals + smoothing)
# -------------------------------
def plot_velocities(t, lin_vel, ang_vel,
                    v_goal=None, w_goal=None,
                    window=20):
    t = np.asarray(t)
    lin_vel = np.asarray(lin_vel)
    ang_vel = np.asarray(ang_vel)

    lin_smooth = rolling_average(lin_vel, window)
    ang_smooth = rolling_average(ang_vel, window)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Linear velocity
    axs[0].plot(t, lin_vel, alpha=0.4, label="Raw")
    axs[0].plot(t, lin_smooth, linewidth=2, label="Rolling Avg")
    # v_goal may be scalar or array
    if v_goal is not None:
        if np.ndim(v_goal) == 0:
            axs[0].axhline(v_goal, linestyle='--', linewidth=2, label="V Goal")
        else:
            axs[0].plot(t, np.asarray(v_goal), linestyle='--', linewidth=2, label="V Goal")

    axs[0].set_title("Linear Velocity vs Time")
    axs[0].set_ylabel("Linear Velocity [mm/s]")
    axs[0].grid(True)
    axs[0].legend()

    # Angular velocity
    axs[1].plot(t, ang_vel, alpha=0.4, label="Raw")
    axs[1].plot(t, ang_smooth, linewidth=2, label="Rolling Avg")
    if w_goal is not None:
        if np.ndim(w_goal) == 0:
            axs[1].axhline(w_goal, linestyle='--', linewidth=2, label="W Goal")
        else:
            axs[1].plot(t, np.asarray(w_goal), linestyle='--', linewidth=2, label="W Goal")

    axs[1].set_title("Angular Velocity vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular Velocity [rad/s]")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    # plt.show()

# -------------------------------
# 4) Acceleration Plots (with smoothing)
# -------------------------------
def plot_accelerations(t, lin_acc, ang_acc, lin_acc_des=None, ang_acc_des=None, window=25):
    """
    Plot linear and angular accelerations (measured) and optional desired arrays/scalars.
    """
    t = np.asarray(t)
    lin_acc = np.asarray(lin_acc)
    ang_acc = np.asarray(ang_acc)

    lin_smooth = rolling_average(lin_acc, window)
    ang_smooth = rolling_average(ang_acc, window)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(t, lin_acc, alpha=0.4, label="Raw")
    axs[0].plot(t, lin_smooth, linewidth=2, label="Rolling Avg")
    if lin_acc_des is not None:
        if np.ndim(lin_acc_des) == 0:
            axs[0].axhline(lin_acc_des, linestyle='--', linewidth=2, label="a_des")
        else:
            axs[0].plot(t, np.asarray(lin_acc_des), linestyle='--', linewidth=2, label="a_des")
    axs[0].set_title("Linear Acceleration vs Time")
    axs[0].set_ylabel("Linear Acceleration [mm/s²]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, ang_acc, alpha=0.4, label="Raw")
    axs[1].plot(t, ang_smooth, linewidth=2, label="Rolling Avg")
    if ang_acc_des is not None:
        if np.ndim(ang_acc_des) == 0:
            axs[1].axhline(ang_acc_des, linestyle='--', linewidth=2, label="ang a_des")
        else:
            axs[1].plot(t, np.asarray(ang_acc_des), linestyle='--', linewidth=2, label="ang a_des")
    axs[1].set_title("Angular Acceleration vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular Acceleration [rad/s²]")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    # plt.show()

# -------------------------------
# 5) Combined plot: Linear Accel (meas vs desired) and Angular Velocity (meas vs desired)
# -------------------------------
def plot_aw(t, lin_acc_meas, ang_vel_meas, lin_acc_des=None, ang_vel_des=None, window=25):
    """
    Top subplot: linear acceleration (measured vs desired)
    Bottom subplot: angular velocity (measured vs desired)

    lin_acc_des and ang_vel_des may be scalar values or arrays matching t.
    """
    t = np.asarray(t)
    lin_acc_meas = np.asarray(lin_acc_meas)
    ang_vel_meas = np.asarray(ang_vel_meas)

    # smoothing
    lin_acc_smooth = rolling_average(lin_acc_meas, window)
    ang_vel_smooth = rolling_average(ang_vel_meas, window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Linear acceleration subplot
    axes[0].plot(t, lin_acc_meas, alpha=0.35, label="a_meas (raw)")
    axes[0].plot(t, lin_acc_smooth, linewidth=1.6, label="a_meas (rolling avg)")
    if lin_acc_des is not None:
        if np.ndim(lin_acc_des) == 0:
            axes[0].axhline(lin_acc_des, linestyle='--', linewidth=2, label="a_des")
        else:
            axes[0].plot(t, np.asarray(lin_acc_des), linestyle='--', linewidth=2, label="a_des")
    axes[0].set_title("Linear Acceleration vs Time")
    axes[0].set_ylabel("Linear Acceleration [mm/s²]")
    axes[0].grid(True)
    axes[0].legend(loc='upper right')

    # Angular velocity subplot
    axes[1].plot(t, ang_vel_meas, alpha=0.35, label="w_meas (raw)")
    axes[1].plot(t, ang_vel_smooth, linewidth=1.6, label="w_meas (rolling avg)")
    if ang_vel_des is not None:
        if np.ndim(ang_vel_des) == 0:
            axes[1].axhline(ang_vel_des, linestyle='--', linewidth=2, label="w_des")
        else:
            axes[1].plot(t, np.asarray(ang_vel_des), linestyle='--', linewidth=2, label="w_des")
    axes[1].set_title("Angular Velocity vs Time")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angular Velocity [rad/s]")
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    # plt.show()

# -------------------------------
# Example usage (uncomment to run)
# -------------------------------
# T = 6.0
# fs = 50.0
# n = int(T * fs)
# t = np.linspace(0, T, n)
# a_des = np.zeros(n); a_des[50:150] = 400
# a_meas = np.roll(a_des, 3) + np.random.randn(n)*60
# w_des = np.zeros(n); w_des[120:300] = 0.8
# w_meas = np.roll(w_des, 2) + np.random.randn(n)*0.02
# plot_accel_and_angvel_with_desired(t, a_meas, a_des, w_meas, w_des, window=15)
