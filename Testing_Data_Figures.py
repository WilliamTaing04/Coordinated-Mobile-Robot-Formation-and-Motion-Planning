import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import Data_Visualization as plot

# Directory where this file lives
HERE = Path(__file__).parent

def save_to_pickle(data: dict, filename: str):
    """Save data dictionary to a pickle file in the same folder as this script."""
    path = HERE / filename
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_from_pickle(filename: str):
    """Load data dictionary from a pickle file in the same folder as this script."""
    path = HERE / filename
    with open(path, "rb") as f:
        return pickle.load(f)

def plots():
    # ALL TEST DATA PATH
    data_circle = Path("TestingData") / "Safe_Formation_Controller_Circle.pkl"
    data_disturb = Path("TestingData") / "Safe_Formation_Controller_Disturbance.pkl"
    data_snake = Path("TestingData") / "Safe_Formation_Controller_Snake.pkl"
    data_static_obs1 = Path("TestingData") / "Obstacle_Avoidance1.pkl"
    data_static_obs2 = Path("TestingData") / "Obstacle_Avoidance2.pkl"
    data_dynamic_obs1 = Path("TestingData") / "Dynamic_Obstacle_Avoidance1.pkl"
    data_dynamic_obs2 = Path("TestingData") / "Dynamic_Obstacle_Avoidance2.pkl"

    # LOAD FROM PICKLE
    data = load_from_pickle(data_circle)

    # EXTRACT DATA FORM DICTIONARY
    t = data["time"]
    pos = data["pos"]
    pos_f = data["pos_f"]
    lin_vel = data["lin_vel"]
    ang_vel = data["ang_vel"]
    lin_vel_f = data["lin_vel_f"]
    ang_vel_f = data["ang_vel_f"]
    lin_acc = data["lin_acc"]
    ang_acc = data["ang_acc"]
    lin_acc_des = data["lin_acc_des"]
    ang_vel_des = data["ang_vel_des"]
    data_long_sb = data["long_sb"]
    data_lat_sb = data["lat_sb"]
    data_long_des = data["long_des"]
    data_lat_des = data["lat_des"]
    data_long_safe_limit = data["long_safe_limit"]
    data_lat_safe_limit = data["lat_safe_limit"]
    data_leader_pos_est = data["leader_pos_est"]
    data_leader_vel_est = data["leader_vel_est"]
    data_form_dist_along = data['form_dist_along']
    data_form_dist_perp = data['form_dist_perp']
    num_bots = pos_f.shape[1]


    # MULTIAGENT PLOTS:
    # # plot.analyze_dt_histogram(t, bins=30, title="dt Histogram")
    plot.plot_all_xy_trajectories(pos_f, labels=["Leader", "Follower 1", "Follower 2", "Obstable 1", "Obstable 2", "Obstable 3"], show_start_end=True)
    plot.basic_plot(t, lin_vel_f[:,:3], "Time (s)", "Linear Velocity (m/s)", ["Leader", "Follower 1", "Follower 2"]) 
    plot.basic_plot(t, ang_vel_f[:,:3], "Time (s)", "Angular Velocity (rad/s)", ["Leader", "Follower 1", "Follower 2"]) 
    plot.basic_plot(t, lin_acc[:,:3], "Time (s)", "Linear Accelertaion (m/s^2)", ["Leader", "Follower 1", "Follower 2"]) 

    # Leader’s position (left) and velocity (right) estimation error when estimated by follower agents
    plot.basic_plot(t, data_leader_pos_est, "Time (s)", "Position Estimate Error (m)", ["Estimate of Leader by Follower 1", "Estimate of Leader by Follower 2"])
    plot.basic_plot(t, data_leader_vel_est, "Time (s)", "Velocity Estimate Error (m/s)", ["Estimate of Leader by Follower 1", "Estimate of Leader by Follower 2"])

    # (d) Distance along the motion, (e) Distance perpendicular to the motion
    plot.basic_plot(t, [data_form_dist_along[:,1], data_form_dist_along[:,2], data_long_des[:,1], data_long_safe_limit[:,1]], "Time (s)", "Formation distance along motion (m)", ["Follower 1","Follower 2", "Desired distance", "Safety limit"])
    # plot.basic_plot(t, [data_form_dist_perp[:,1], data_form_dist_perp[:,2], data_lat_des[:,1], data_lat_des[:,2], data_lat_safe_limit[:,1], data_lat_safe_limit[:,2]], "Time (s)", "Formation distance perpendicular to motion (m)", ["Follower 1","Follower 2", "Desired distance 1","Desired distance 2", "Safety limit1", "Safety limit2"])
    plot.basic_plot(t, [abs(data_form_dist_perp[:,1]), data_form_dist_perp[:,2], data_lat_des[:,2],data_lat_safe_limit[:,2],], "Time (s)", "Formation distance perpendicular to motion (m)", ["Follower 1","Follower 2","Desired distance", "Safety limit"])
    
    # (c) Longitudinal safety barrier function evolution
    plot.basic_plot(t, data_long_sb[:,1:3], "Time (s)", "Safety along motion (m)", ["Follower 1", "Follower 2"])
    # (d) Lateral safety barrier function evolution for static/dynamic obstacles
    plot.basic_plot(t, data_lat_sb[:,1:3], "Time (s)", "Safety perpendicular to motion (m)", ["Follower 1", "Follower 2"])

    plt.show()


    # Per agent individual plots
    # for i in range(3):
        # plot.plot_xy_trajectory(pos_f[:, i, :], title=f"Robot {i} XY Trajectory", show_start_end=True)
        # plot.plot_pose_raw_vs_filtered(t, pose_raw=pos[:, i, :], pose_filt=pos_f[:, i, :], title=f"Robot {i} Pose: Raw vs Filtered")
        # plot.plot_xy_vs_time(t, pos_f[:, i, :], title=f"Robot {i} Position vs Time (Filtered)")
        # plot.plot_velocity_raw_vs_filtered(t, lin_vel[:, i], ang_vel[:, i], lin_vel_f[:, i], ang_vel_f[:, i], title=f"Robot {i} Velocities: Raw vs Filtered")
        # plot.plot_velocities(t, lin_vel_f[:, i], ang_vel_f[:, i], v_des=None, w_des=ang_vel_des[:, i], title=f"Robot {i} Velocities vs Time (Filtered)")
        # plot.plot_accelerations(t, lin_acc[:, i], ang_acc[:, i], a_des=lin_acc_des[:, i], title=f"Robot {i} Accelerations vs Time", window=30, plot_raw=True)
        # plot.plot_accel_and_angvel(t, lin_acc[:, i], ang_vel_f[:, i], lin_acc_des[:, i], ang_vel_des[:, i], title=f"Robot {i} UW actual vs desired")
    # plt.show()


if __name__ == "__main__":
    # main()
    plots()