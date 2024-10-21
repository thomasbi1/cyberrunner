#ifndef DYNAMIXEL_CONTROLLER_H
#define DYNAMIXEL_CONTROLLER_H

int dynamixel_init(const char* port, int num_dynamixel, int* dynamixel_ids, int baudrate, uint16_t max_temp, uint32_t* positions);
int dynamixel_step(int num_dynamixel, int* dynamixel_ids, int32_t* moving_speed);

#endif
