#include <fcntl.h>
#include <termios.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include "dynamixel_sdk/dynamixel_sdk.h"

#include "cyberrunner_dynamixel/dynamixel_controller.h"
#include "cyberrunner_dynamixel/error_handling.h"

// Control table address, these might be different for different models.
#define ADDR_XL_TORQUE_ENABLE           64
#define ADDR_XL_PRESENT_VELOCITY        128
#define ADDR_XL_TEMPERATURE_LIMIT       31
#define ADDR_XL_PRESENT_POSITION        132
#define ADDR_XL_VELOCITY_KI             76
#define ADDR_XL_VELOCITY_KP             78
#define ADDR_XL_GOAL_CURRENT            102
#define ADDR_XL_GOAL_VELOCITY           104
#define ADDR_XL_HARDWARE_ERROR_STATUS   70  // Hardware error packet retrieval
#define ADDR_XL_PWM_OUTPUT              100 // Address for PWM output

// Protocol version
#define PROTOCOL_VERSION                2.0 // See which protocol version is used in the Dynamixel

#define XL_TEMPERATURE_LIMIT            50  // Maximum Temperature Allowable for Dynamixel
#define TORQUE_ENABLE                   1   // Value for enabling the torque
#define TORQUE_DISABLE                  0   // Value for disabling the torque
#define I_GAIN_VALUE                    400
#define P_GAIN_VALUE                    40

dynamixel::PortHandler *port_handler = NULL;
dynamixel::PacketHandler *packet_handler = NULL;

using namespace dynamixel;

int dynamixel_init(const char* port, int num_dynamixel, int* dynamixel_ids, int baudrate, uint16_t max_temp, uint32_t* positions)
{
    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_error = 0;

    if(port_handler == NULL || packet_handler == NULL)
    {
        // Minimize latency on USB communication TODO: wrap in shell script
        char device[10];
        char cmd[256];
        sscanf(port, "/dev/%s", device);
        sprintf(cmd, "echo 1 | sudo tee /sys/bus/usb-serial/devices/%s/latency_timer", device);
        printf("Executing command: %s\n", cmd);
        system(cmd);

        // Initialize PortHandler Structs
        port_handler = PortHandler::getPortHandler(port);

        // Initialize PacketHandler Structs
        packet_handler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        // Open port
        if (port_handler->openPort())
        {
            printf("Succeeded to open the port!\n");
        }
        else
        {
            printf("Failed to open the port!\n");
            return -1;
        }

        // Set port baudrate
        if (port_handler->setBaudRate(baudrate))
        {
            printf("Succeeded to change the baudrate!\n");
        }
        else
        {
            printf("Failed to change the baudrate!\n");
            return -1;
        }
    }

    for(int i = 0; i < num_dynamixel; i++)
    {
        // Dis- and enable Dynamixel Torque
        dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to disable torque for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            return -1;
        }
        if (dxl_error != 0)
        {
            printf("Error disabling torque for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            return -1;
        }

        dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to enable torque for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            return -1;
        }
        if (dxl_error != 0)
        {
            printf("Error enabling torque for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            return -1;
        }
        
        printf("Dynamixel (ID: %d) has been successfully connected \n", dynamixel_ids[i]);

        // Read present position
        dxl_comm_result = packet_handler->read4ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_PRESENT_POSITION, (uint32_t*)&(positions[i]), &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to read present position for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            return -1;
        }
        if (dxl_error != 0)
        {
            printf("Error reading present position for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            return -1;
        }
    }

    return 0;
}

void print_hardware_error_status(int dynamixel_id)
{
    uint8_t hardware_error_status = 0;
    uint8_t dxl_error = 0;
    int dxl_comm_result = packet_handler->read1ByteTxRx(port_handler, dynamixel_id, ADDR_XL_HARDWARE_ERROR_STATUS, &hardware_error_status, &dxl_error);

    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to read hardware error status for Dynamixel ID %d: %s\n", dynamixel_id, packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Hardware error status for Dynamixel ID %d: %s\n", dynamixel_id, packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Hardware error status for Dynamixel ID %d: 0x%02X\n", dynamixel_id, hardware_error_status);
    }
}

int dynamixel_step(int num_dynamixel, int* dynamixel_ids, int32_t* moving_speed)
{
    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_error = 0;

    for(int i = 0; i < num_dynamixel; i++)
    {
        // Read current PWM output
        /*
        int16_t pwm_output = 0;
        dxl_comm_result = packet_handler->read2ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_PWM_OUTPUT, (uint16_t*)&pwm_output, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to read PWM output for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            continue;
        }
        if (dxl_error != 0)
        {
            printf("Error reading PWM output for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            continue;
        }
 

        // Debugging statement
        
        //printf("Dynamixel ID %d: Current PWM output = %d\n", dynamixel_ids[i], pwm_output);

        // Write goal velocity (4 bytes)
        int32_t goal_velocity = moving_speed[i];
        dxl_comm_result = packet_handler->write4ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_GOAL_VELOCITY, goal_velocity, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to write goal velocity for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            continue;
        }
        if (dxl_error != 0)
        {
            printf("Error writing goal velocity for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            print_hardware_error_status(dynamixel_ids[i]);

            // Check for overload error and handle it
            uint8_t hardware_error_status = 0;
            packet_handler->read1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_HARDWARE_ERROR_STATUS, &hardware_error_status, &dxl_error);
            if (hardware_error_status & 0x10) // Bit 4 indicates overload error
            {
                printf("Overload error detected on Dynamixel ID %d. Disabling torque.\n", dynamixel_ids[i]);
                packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
                continue;
            }
        }
        */
        // Write goal current (2 bytes)
        int16_t goal_current = moving_speed[i];
        dxl_comm_result = packet_handler->write2ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_GOAL_CURRENT, goal_current, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to write goal current for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            continue;
        }
        if (dxl_error != 0)
        {
            printf("Error writing goal current for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            print_hardware_error_status(dynamixel_ids[i]);

            // Check for overload error and handle it
            uint8_t hardware_error_status = 0;
            packet_handler->read1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_HARDWARE_ERROR_STATUS, &hardware_error_status, &dxl_error);
            if (hardware_error_status & 0x10) // Bit 4 indicates overload error
            {
                printf("Overload error detected on Dynamixel ID %d. Disabling torque.\n", dynamixel_ids[i]);
                packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
                continue;
            }
        }
    }

    return 0;
}
