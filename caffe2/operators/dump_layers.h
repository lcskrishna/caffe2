#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>

#ifndef ENABLE_DUMP_LAYERS
	#define ENABLE_DUMP_LAYERS 1
#endif

static int layer_count = 0;

void increment_layer_count();
int get_layer_count();

