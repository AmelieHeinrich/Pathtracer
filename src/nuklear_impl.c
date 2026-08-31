// The single translation unit holding nuklear's implementation. It is built on its own with
// warnings disabled (see xmake.lua): nuklear.h does not compile clean under -Wall, and that
// is not a reason to lower the bar for the rest of the project.
#define NK_IMPLEMENTATION
#include "nuklear_config.h"
