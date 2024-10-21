Train
=====

1. Open four terminal windows. For each, navigate to the CyberRunner workspace and source the workspace.

        cd cyberrunner_ws
        source install/setup.bash

2. In terminal 1 run

        ros2 run cyberrunner_camera cam_publisher.py

3. In terminal 2 run

        ros2 run cyberrunner_state_estimation estimator_sub

4. In terminal 3 run 

        ros2 run cyberrunner_dynamixel cyberrunner_dynamixel

5. In terminal 4 run

        ros2 run cyberrunner_dreamer train
        
    and wait until you see the first logs for step 0.

6. Place the ball at the starting location on the labyrinth. CyberRunner will start playing the game.
