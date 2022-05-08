#ifndef CPP_EXAMPLE_MENU_H
#define CPP_EXAMPLE_MENU_H

#include "arrus/core/api/arrus.h"
#include "imaging/pwi.h"

/**
 * Runs main menu loop (asks user for input, applies new settings, etc.).
 */
void runMainMenu(::arrus::devices::Us4R* us4r, const ::arrus_example_imaging::PwiSequence &seq);

#endif //CPP_EXAMPLE_MENU_H
