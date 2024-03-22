#define DISTRIBUTE_PRINT(message) if(world.rank() == 0) {std::cout<<message<<std::endl;}
