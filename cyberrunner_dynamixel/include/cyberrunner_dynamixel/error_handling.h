#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

// TODO use ros_error
#define THROW(format, ...) \
    do{fprintf(stderr, "%s() at %s:%d: ", __FUNCTION__, __FILE__, __LINE__); \
    fprintf(stderr, format, ##__VA_ARGS__); fprintf(stderr, "\n"); return false;} while(0)

#define ASSERT(expr) \
    do{if(!expr) {fprintf(stderr, "%s() at %s:%d", __FUNCTION__, __FILE__, __LINE__); return false;}} while(0)

#endif
