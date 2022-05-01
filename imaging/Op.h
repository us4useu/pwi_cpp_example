#ifndef CPP_EXAMPLE_IMAGING_OP_H
#define CPP_EXAMPLE_IMAGING_OP_H

#include <string>

namespace imaging {

class Op {
public:
    using OpClassId = std::string;

    virtual OpClassId getOpClassId() = 0;

private:
    OpClassId id;
};

}

#endif//CPP_EXAMPLE_IMAGING_OP_H
