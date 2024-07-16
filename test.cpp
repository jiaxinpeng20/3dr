
#include <boost/mpi.hpp>
#include <iostream>
#include <vector>

// Define a complex data type
struct Data {
    int id;
    std::string name;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & id;
        ar & name;
    }
};

int main() {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<Data> sendBuffer;
    Data data;
    data.id = world.rank();
    data.name = "Process " + std::to_string(world.rank());
    sendBuffer.push_back(data);

    std::vector<int> recvCounts;
    boost::mpi::gather(world, sendBuffer.size(), recvCounts, 0);

    std::vector<Data> recvBuffer;
    boost::mpi::gatherv(world, sendBuffer, recvBuffer, recvCounts, 0);

    if (world.rank() == 0) {
        std::cout << "Received data:\n";
        for (const auto& recvData : recvBuffer) {
            std::cout << "ID: " << recvData.id << ", Name: " << recvData.name << std::endl;
        }
    }

    return 0;
}

