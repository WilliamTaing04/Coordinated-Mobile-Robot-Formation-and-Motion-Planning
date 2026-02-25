import numpy as np
import matplotlib.pyplot as plt

def _as_desired_array(desired, t):
    """
    Convert desired input (None | scalar | array-like | callable) to an array aligned with t.
    """
    if desired is None:
        return None

    t = np.asarray(t)

    if callable(desired):
        y = np.asarray(desired(t), dtype=float)
        if y.shape[0] != t.shape[0]:
            raise ValueError("Callable desired(t) must return array of same length as t.")
        return y

    if np.ndim(desired) == 0:
        return np.full_like(t, float(desired), dtype=float)

    y = np.asarray(desired, dtype=float)
    if y.shape[0] != t.shape[0]:
        raise ValueError("Desired array must have same length as t.")
    return y

def rolling_average(x, window):
    """
    Simple windowed rolling average (centered) using convolution.
    window <= 1 disables smoothing and returns original array.
    """
    x = np.asarray(x, dtype=float)
    if window is None or int(window) <= 1:
        return x
    w = int(window)
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


# -------------------------------
# XY Trajectory
# -------------------------------
def plot_xy_trajectory(
    pose,
    title="X vs Y Trajectory",
    label=None,
    show_start_end=True,
):
    pose = np.asarray(pose)
    x = pose[:, 0]
    y = pose[:, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label=label)

    if show_start_end and len(x) > 0:
        plt.scatter(x[0], y[0], marker="o", label="Start")
        plt.scatter(x[-1], y[-1], marker="x", label="End")

    plt.title(title)
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.axis("equal")
    plt.grid(True)
    if label is not None or show_start_end:
        plt.legend()
    plt.tight_layout()


# -------------------------------
# X(t) and Y(t)
# -------------------------------
def plot_xy_vs_time(
    t,
    pose,
    title="Position vs Time",
):
    t = np.asarray(t)
    pose = np.asarray(pose)

    x = pose[:, 0]
    y = pose[:, 1]

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axs[0].plot(t, x)
    axs[0].set_title("X vs Time")
    axs[0].set_ylabel("X [mm]")
    axs[0].grid(True)

    axs[1].plot(t, y)
    axs[1].set_title("Y vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Y [mm]")
    axs[1].grid(True)

    fig.suptitle(title)
    fig.tight_layout()


# -------------------------------
# Pose Raw vs Filtered
# -------------------------------
def plot_pose_raw_vs_filtered(
    t,
    pose_raw,
    pose_filt,
    title="Pose: Raw vs Filtered",
):
    t = np.asarray(t)
    pr = np.asarray(pose_raw)
    pf = np.asarray(pose_filt)

    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    # X
    axs[0].plot(t, pr[:, 0], label="Raw")
    axs[0].plot(t, pf[:, 0], label="Filtered")
    axs[0].set_ylabel("X [mm]")
    axs[0].grid(True)
    axs[0].legend()

    # Y
    axs[1].plot(t, pr[:, 1], label="Raw")
    axs[1].plot(t, pf[:, 1], label="Filtered")
    axs[1].set_ylabel("Y [mm]")
    axs[1].grid(True)
    axs[1].legend()

    # Yaw
    axs[2].plot(t, pr[:, 2], label="Raw")
    axs[2].plot(t, pf[:, 2], label="Filtered")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Yaw [rad]")
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle(title)
    fig.tight_layout()

# -------------------------------
# Velocity Raw vs Filtered
# -------------------------------
def plot_velocity_raw_vs_filtered(
    t,
    lin_vel_raw,
    ang_vel_raw,
    lin_vel_filt,
    ang_vel_filt,
    title="Velocity: Raw vs Filtered",
):
    t = np.asarray(t)
    v_r = np.asarray(lin_vel_raw)
    w_r = np.asarray(ang_vel_raw)
    v_f = np.asarray(lin_vel_filt)
    w_f = np.asarray(ang_vel_filt)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axs[0].plot(t, v_r, label="Raw")
    axs[0].plot(t, v_f, label="Filtered")
    axs[0].set_title("Linear Velocity")
    axs[0].set_ylabel("v [mm/s]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, w_r, label="Raw")
    axs[1].plot(t, w_f, label="Filtered")
    axs[1].set_title("Angular Velocity")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("w [rad/s]")
    axs[1].grid(True)
    axs[1].legend()

    fig.suptitle(title)
    fig.tight_layout()

# -------------------------------
# Velocities vs Time
# -------------------------------
def plot_velocities(
    t,
    lin_vel,
    ang_vel,
    v_des=None,
    w_des=None,
    title="Velocities vs Time",
):
    t = np.asarray(t)
    lin_vel = np.asarray(lin_vel)
    ang_vel = np.asarray(ang_vel)

    v_des_arr = _as_desired_array(v_des, t)
    w_des_arr = _as_desired_array(w_des, t)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Linear velocity
    axs[0].plot(t, lin_vel, label="Measured")
    if v_des_arr is not None:
        axs[0].plot(t, v_des_arr, linestyle="--", label="Desired")
    axs[0].set_title("Linear Velocity")
    axs[0].set_ylabel("v [mm/s]")
    axs[0].grid(True)
    axs[0].legend()

    # Angular velocity
    axs[1].plot(t, ang_vel, label="Measured")
    if w_des_arr is not None:
        axs[1].plot(t, w_des_arr, linestyle="--", label="Desired")
    axs[1].set_title("Angular Velocity")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("w [rad/s]")
    axs[1].grid(True)
    axs[1].legend()

    fig.suptitle(title)
    fig.tight_layout()


# -------------------------------
# Accelerations vs Time
# -------------------------------
# -------------------------------
# Accelerations vs Time (with optional rolling average)
# -------------------------------
def plot_accelerations(
    t,
    lin_acc,
    ang_acc,
    a_des=None,
    alpha_des=None,
    title="Accelerations vs Time",
    window=1,
    plot_raw=True,
):
    """
    window: rolling average window size (samples). window<=1 disables.
    plot_raw: if True, plot raw + smoothed; if False, plot only smoothed.
    """
    t = np.asarray(t)
    lin_acc = np.asarray(lin_acc)
    ang_acc = np.asarray(ang_acc)

    a_des_arr = _as_desired_array(a_des, t)
    alpha_des_arr = _as_desired_array(alpha_des, t)

    lin_acc_s = rolling_average(lin_acc, window)
    ang_acc_s = rolling_average(ang_acc, window)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Linear acceleration
    if plot_raw:
        axs[0].plot(t, lin_acc, alpha=0.35, label="Measured (raw)")
    axs[0].plot(t, lin_acc_s, label=f"Measured (roll avg, w={int(window)})" if int(window) > 1 else "Measured")
    if a_des_arr is not None:
        axs[0].plot(t, a_des_arr, linestyle="--", label="Desired")
    axs[0].set_title("Linear Acceleration")
    axs[0].set_ylabel("a [mm/s²]")
    axs[0].grid(True)
    axs[0].legend()

    # Angular acceleration
    if plot_raw:
        axs[1].plot(t, ang_acc, alpha=0.35, label="Measured (raw)")
    axs[1].plot(t, ang_acc_s, label=f"Measured (roll avg, w={int(window)})" if int(window) > 1 else "Measured")
    if alpha_des_arr is not None:
        axs[1].plot(t, alpha_des_arr, linestyle="--", label="Desired")
    axs[1].set_title("Angular Acceleration")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("α [rad/s²]")
    axs[1].grid(True)
    axs[1].legend()

    fig.suptitle(title)
    fig.tight_layout()