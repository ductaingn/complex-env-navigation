#pragma once
#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim{
    class ENVIRONMENT : public RaisimGymEnv
{
    int main(){
        world_ = std::make_unique<raisim::World>();

    }
}
}

