import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    #Getting the actual params of link 1 to 6
    a_known = np.concatenate([
        dyn_model.pin_model.inertias[i].toDynamicParameters()
        for i in range(1, 7)
    ])
    a_7_known = dyn_model.pin_model.inertias[7].toDynamicParameters()
    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []
    
    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        Y = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        regressor_all.append(Y.copy())
        tau_mes_all.append(tau_mes)

        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")
    print("Shape of regressor all", np.shape(regressor_all))
    print("Shape of tau_mes", np.shape(tau_mes_all))
    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    Y_all = np.vstack(regressor_all)
    tau_all = np.concatenate(tau_mes_all)
    print("Shape of stacked regressor:", np.shape(Y_all))
    print("Shape of stacked tau:", np.shape(tau_all))
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    Y_known = Y_all[:, :60]
    Y_link7 = Y_all[:, 60:70]
    torque_link7 = tau_all - Y_known @ a_known
    a_link7 = np.linalg.pinv(Y_link7) @ torque_link7
    parameter_error = a_link7 - a_7_known
    print("True parameters:", a_7_known)
    print("Estimated_params:", a_link7)
    print("Parameter error:", parameter_error)

    # Evaluate the fitted model on the collected training samples.
    tau_predicted = Y_known @ a_known + Y_link7 @ a_link7
    torque_error = tau_all - tau_predicted
    RMSE = np.sqrt(np.mean(torque_error**2))
    print("Training RMSE:", RMSE)
    tau_true_model = Y_known @ a_known + Y_link7 @ a_7_known
    true_model_rmse = np.sqrt(np.mean((tau_all - tau_true_model)**2))

    print("True-model RMSE:", true_model_rmse)
    # Y_samples = np.stack(regressor_all)
    # print("\nTraining metrics (adjusted R² uses effective regressor rank):")
    # for joint in range(num_joints):
    #     rank = np.linalg.matrix_rank(Y_samples[:, joint, :])
    #     metrics = compute_regression_metrics(
    #         tau_measured[:, joint], tau_predicted[:, joint], rank
    #     )
    #     print(
    #         f"Joint {joint + 1}: RMSE={metrics['rmse']:.4f} N·m, "
    #         f"MAE={metrics['mae']:.4f} N·m, "
    #         f"R²={metrics['r_squared']:.4f}, "
    #         f"adjusted R²={metrics['adjusted_r_squared']:.4f}, rank={rank}"
    #     )

    # # Plot measured minus predicted torque for each joint.
    # torque_error = tau_measured - tau_predicted
    # sample_times = np.arange(len(tau_measured)) * time_step
    # fig, axes = plt.subplots(num_joints, 1, sharex=True,
    #                          figsize=(10, 2 * num_joints), squeeze=False)
    # for joint, ax in enumerate(axes[:, 0]):
    #     ax.plot(sample_times, torque_error[:, joint])
    #     ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    #     ax.set_ylabel(f"Joint {joint + 1}\nError (N·m)")
    #     ax.grid(True)
    # axes[-1, 0].set_xlabel("Simulation time (s)")
    # fig.suptitle("Torque prediction error: measured − predicted")
    # fig.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
