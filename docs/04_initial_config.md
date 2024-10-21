Initial Configuration
=====

Before we can start using CyberRunner, we need to first **calibrate the camera** and **configure the motors**. These steps have to be taken only once. 

## Camera Calibration

In order for the position of the ball within the labyrinth to be tracked, we need to first calibrate the camera. For this purpose, we use the <a href="https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab" target="_blank">COCamCalib MATLAB toolbox</a>.

1. Follow the steps <a href="https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab" target="_blank">here</a> to obtain the calibration results for your camera. 

    !!! warning "Disclaimer"

        The **Find Center** step is computationally intensive and may get stuck. In this case, update `PP` in both instances of the function `lsqlin()` to `sparse(PP)`. This should speed up the process. For more details, the bug fix is idenitified in <a href="https://groups.google.com/g/ocamcalib-toolbox/c/ZgVP4LHd-6w?pli=1" target="_blank">this thread</a>.

    Make sure to take the calibration pictures at a 1920x1200 resolution.

2. Save the calibration results by clicking on **Export Data** in the OCamCalib GUI. This will create the file `calib_results.txt` in your MATLAB workspace.

3. Copy the `calib_results.txt` file as `calib_results_cyberrunner.txt` to your ROS workspace with

        cp <path_to_matlab_workspace>/calib_results.txt <path_to_cyberrunner_ws>/src/cyberrunner/cyberrunner_state_estimation/calib/calib_results_cyberrunner.txt
    Replace `<path_to_matlab_workspace>` and `<path_to_cyberrunner_ws>` with the path to your MATLAB workspace and your CyberRunner ROS workspace, respectively.

4. Install the calibration file by navigating to your workspace and running

        colcon build --symlink-install
        source install/setup.bash

## Marker positions

The state estimator relies on approximately knowing the location of the blue markers within the image. To obtain these positions follow these steps.

1. Open two terminal windows, navigate to the CyberRunner workspace and source the workspace.

2. In the first terminal, run the camera script

        ros2 run cyberrunner_camera cam_publisher.py

3. In the second terminal, run the select markers script and follow the instructions in the terminal

        ros2 run cyberrunner_state_estimation select_markers
        

## Motor Configuration

1. Install [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/).
2. Temporarily disconnect the front motor (the motor on the longer side of the labyrinth). Connect the U2D2 to your host PC and ensure that the motors are turned on by pressing the power switch on the U2D2 Power Hub Board.
3. In the Dynamixel Wizard, press **Scan** until the motor is found.
4. Assign **ID 2** to the connected motor, set the baud rate to 1M, and set the operating mode to 0 (Current Control Mode). Make sure to click **Save** after each change.
5. Reconnect the front motor.
6. Press **Scan** again.
7. For the newly connected motor, assign **ID 1**, and apply the same changes as for the other motor.
