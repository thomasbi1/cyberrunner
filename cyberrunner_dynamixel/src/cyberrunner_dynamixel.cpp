#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <unistd.h>
#include <string.h>

#include <chrono>
#include <thread>


#include "cyberrunner_dynamixel/error_handling.h"
#include "cyberrunner_dynamixel/dynamixel_controller.h"
#include "cyberrunner_interfaces/msg/dynamixel_vel.hpp"
#include "cyberrunner_interfaces/srv/dynamixel_reset.hpp"
#include "rclcpp/rclcpp.hpp"


#define DYNAMIXEL_ID_1 (1)
#define DYNAMIXEL_ID_2 (2)

#define ERR(err) \
    do {exit(EXIT_FAILURE);} while(0)

// TODO: ERROR HANDLING!!!


// TODO clean this up
int32_t positions[2];
const char* port = "/dev/ttyUSB0";


void set_dynamixel_speed(const cyberrunner_interfaces::msg::DynamixelVel::SharedPtr msg)
{
    int32_t dynamixel_ids[2];
    int32_t moving_speeds[2];

    // Moving speeds from message
    moving_speeds[0] = msg->vel_1;
    moving_speeds[1] = msg->vel_2;

    // Dynamixel ids
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;

    dynamixel_step(2, dynamixel_ids, moving_speeds);
}


void reset_dynamixel(const std::shared_ptr<cyberrunner_interfaces::srv::DynamixelReset::Request> request,
          std::shared_ptr<cyberrunner_interfaces::srv::DynamixelReset::Response> response)
{
    int32_t dynamixel_ids[2];
    int32_t moving_speeds[2];

    // Dynamixel ids
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;

    // Reset motors  TODO: return not 0 if failed
    dynamixel_init(port, 2, dynamixel_ids, 1000000, 50, (uint32_t*)positions);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Moving speeds from message
    moving_speeds[0] = -150;
    moving_speeds[1] = 150;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    moving_speeds[0] = 0;
    moving_speeds[1] = 0;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    dynamixel_init(port, 2, dynamixel_ids, 1000000, 50, (uint32_t*)positions);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    moving_speeds[0] = 100;
    moving_speeds[1] = 0;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    moving_speeds[0] = 0;
    moving_speeds[1] = 0;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    dynamixel_init(port, 2, dynamixel_ids, 1000000, request->max_temp, (uint32_t*)positions);

    response->success = 1;
}


int main(int argc, char** argv)
{
    // Init ROS
    rclcpp::init(argc, argv);

    // Get device name
    if(argc >= 2) port = argv[1];
    printf("Using port: %s. Use the command-line argument to change the port, e.g.:\n  ros2 run cyberrunner_dynamixel cyberrunner_dynamixel /dev/ttyUSB*\n", port);

    // Initialize dynamixel
    int32_t dynamixel_ids[2];
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;
    dynamixel_init(port, 2, dynamixel_ids, 1000000, 256, (uint32_t*)positions);  //TODO make param file

    rclcpp::NodeOptions options;
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("cyberrunner_dynamixel", options);
    rclcpp::Subscription<cyberrunner_interfaces::msg::DynamixelVel>::SharedPtr sub = node->create_subscription<cyberrunner_interfaces::msg::DynamixelVel>("cyberrunner_dynamixel/cmd", 1, &set_dynamixel_speed);
    rclcpp::Service<cyberrunner_interfaces::srv::DynamixelReset>::SharedPtr service = node->create_service<cyberrunner_interfaces::srv::DynamixelReset>("cyberrunner_dynamixel/reset", &reset_dynamixel);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
