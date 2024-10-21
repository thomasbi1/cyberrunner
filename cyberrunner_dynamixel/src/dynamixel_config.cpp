#include <stdio.h>
#include "dynamixel_sdk/dynamixel_sdk.h"

// ADDRESSES
#define ADDR_ID                 7
#define ADDR_BAUD_RATE          8
#define ADDR_OPERATING_MODE     11
#define ADDR_PROTOCOL_TYPE      13
#define ADDR_TEMPERATURE_LIMIT  31
#define ADDR_PWM_LIMIT          36
#define ADDR_SHUTDOWN           63
#define ADDR_VELOCITY_I		76
#define ADDR_VELOCITY_P		78

// VALUES
#define ID_1                    1
#define ID_2                    2
#define BAUD_RATE               3
#define OPERATING_MODE          0
#define PROTOCOL_TYPE           2
#define TEMPERATURE_LIMIT       70
#define PWM_LIMIT               666
#define SHUTDOWN                0x15
#define VELOCITY_I              400
#define VELOCITY_P              40


uint32_t baudrates[7] = {9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000};


void configure_dynamixel(dynamixel::PortHandler* port_handler, dynamixel::PacketHandler* packet_handler, uint8_t dxl_id, uint8_t new_id)
{
    int dxl_comm_result;
    uint8_t dxl_error;

    // Set ID
    dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, dxl_id, ADDR_ID, new_id, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set ID: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting ID: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("ID set to %d successfully\n", new_id);
    }

    // Set Baud Rate
    dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, new_id, ADDR_BAUD_RATE, BAUD_RATE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Baud Rate: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Baud Rate: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Baud Rate set successfully\n");
    }

    // Set Operating Mode
    dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, new_id, ADDR_OPERATING_MODE, OPERATING_MODE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Operating Mode: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Operating Mode: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Operating Mode set successfully\n");
    }

    // Set Protocol Type
    dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, new_id, ADDR_PROTOCOL_TYPE, PROTOCOL_TYPE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Protocol Type: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Protocol Type: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Protocol Type set successfully\n");
    }

    // Set Temperature Limit
    dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, new_id, ADDR_TEMPERATURE_LIMIT, TEMPERATURE_LIMIT, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Temperature Limit: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Temperature Limit: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Temperature Limit set successfully\n");
    }

    // Set PWM Limit
    dxl_comm_result = packet_handler->write2ByteTxRx(port_handler, new_id, ADDR_PWM_LIMIT, PWM_LIMIT, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set PWM Limit: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting PWM Limit: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("PWM Limit set successfully\n");
    }

    // Set Shutdown value
    dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, new_id, ADDR_SHUTDOWN, SHUTDOWN, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Shutdown: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Shutdown: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Shutdown value set successfully\n");
    }

    // Set Velocity I Gain
    dxl_comm_result = packet_handler->write2ByteTxRx(port_handler, new_id, ADDR_VELOCITY_I, VELOCITY_I, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Velocity I Gain: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Velocity I Gain: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Velocity I Gain set successfully\n");
    }

    // Set Velocity P Gain
    dxl_comm_result = packet_handler->write2ByteTxRx(port_handler, new_id, ADDR_VELOCITY_P, VELOCITY_P, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to set Velocity P Gain: %s\n", packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Error setting Velocity P Gain: %s\n", packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Velocity P Gain set successfully\n");
    }
}

uint8_t scan_for_dynamixel(dynamixel::PortHandler* port_handler, dynamixel::PacketHandler* packet_handler)
{
    int dxl_comm_result;
    uint8_t dxl_error;
    uint8_t found_id = 0;

    for (uint8_t id = 1; id < 253; ++id)
    {
        dxl_comm_result = packet_handler->ping(port_handler, id, &dxl_error);
        if (dxl_comm_result == COMM_SUCCESS)
        {
            printf("Dynamixel with ID %d found\n", id);
            found_id = id;
            break;
        }
    }

    return found_id;
}

int main(int argc, char *argv[])
{
    // Get device
    const char* port = "/dev/ttyUSB0";
    if(argc >= 2) port = argv[1];
    printf("Using port: %s. Use the command-line argument to change the port, e.g.:\n  ros2 run cyberrunner_dynamixel dynamixel_config /dev/ttyUSB*\n", port);

    // Initialize PortHandler instance
    dynamixel::PortHandler *port_handler = dynamixel::PortHandler::getPortHandler(port);

    // Initialize PacketHandler instance
    dynamixel::PacketHandler *packet_handler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_TYPE);

    // Open port
    if (!port_handler->openPort())
    {
        printf("Failed to open the port!\n");
        return 0;
    }
    printf("Succeeded to open the port!\n");

    // Set port baudrate
    if (!port_handler->setBaudRate(1000000))
    {
        printf("Failed to change the baudrate!\n");
        return 0;
    }
    printf("Succeeded to change the baudrate!\n");

    printf("Plug in the Dynamixel to be assigned ID 1 [CONTROLS TILT OF LONGER EDGE OF MAZE] and press Enter...\n");
    getchar();  // Wait for user input

    // Scan for first Dynamixel
    uint8_t current_id = scan_for_dynamixel(port_handler, packet_handler);
    if (current_id == 0)
    {
        printf("No Dynamixel found. Please check the connection.\n");
        return 0;
    }

    // Configure first Dynamixel
    configure_dynamixel(port_handler, packet_handler, current_id, ID_1);

    printf("Unplug current Dynamixel and plug in the Dynamixel to be assigned ID 2 [CONTROLS TILT OF SHORTER EDGE OF MAZE] and press Enter...\n");
    getchar();  // Wait for user input

    // Scan for second Dynamixel
    current_id = scan_for_dynamixel(port_handler, packet_handler);
    if (current_id == 0)
    {
        printf("No Dynamixel found. Please check the connection.\n");
        return 0;
    }

    // Configure second Dynamixel
    configure_dynamixel(port_handler, packet_handler, current_id, ID_2);

    // Close port
    port_handler->closePort();

    return 0;
}
