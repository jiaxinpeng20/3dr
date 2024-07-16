#include "mpi.h"

//Typename T mush be standard C++ data types

namespace XSFM
{
    namespace Communication
    {
        template<typename T>
        void AllGather(const boost::mpi::communicator& world, std::vector<Eigen::MatrixXd>& inData, std::vector<Eigen::MatrixXd>& outData)
        {
            
        }
        
        //Gather all complex data to root process
        template<typename T>
        void Gatherv(boost::mpi::communicator& world, std::vector<std::pair<std::pair<T, T>, std::vector<std::pair<T, T>>>>& inData, std::vector<std::pair<std::pair<T, T>, std::vector<std::pair<T, T>>>>& outData, int root)
        {
            //Get the skeletons of complex data in parallel
            std::vector<int> itemDataCountList;
            for(auto iter = inData.begin(); iter != inData.end(); iter++)
            {
                itemDataCountList.push_back(iter->second().size() * 2 + 2);
            }
            
            int localDataCount = std::accumulate(itemDataCountList.begin(), itemDataCountList.end(), 0);
            int itemCount = inData.size();
            
            
            //Serialize complex data structures
            std::vector<T> serializedData(localDataCount);
            int _index = 0;
            for(int i = 0; i < inData.size(); i++)
            {
                serializedData.at(_index++) = inData.at(i).first().first();
                serializedData.at(_index++) = inData.at(i).first().second();
                
                for(int j = 0; j < inData.at(i).second.size(); j++)
                {
                    serializedData.at(_index++) = inData.at(i).second().at(j).first();
                    serializedData.at(_index++) = inData.at(i).second().at(j).second();
                }
            }
            
            
            
            
            if(world.rank() == root)//root node
            {              
                //Gather the skeletons of complex data from all computing nodes 
                std::vector<int> gatherItemCountList(world.size());   
                boost::mpi::gather(world, itemCount, gatherItemCountList, root);//gather all item counts    
                int globalItemCount = std::accumulate(gatherItemCountList.begin(), gatherItemCountList.end(), 0);
                std::vector<int> gatherItemDataCountList(globalItemCount);
                boost::mpi::gatherv(world, itemDataCountList, gatherItemDataCountList.data(), gatherItemCountList, root);
                
                
                //Gather the content of complex data from all computing nodes
                std::vector<int> gatherDataCountList;
                int globalDataCount;
                boost::mpi::reduce(world, localDataCount, globalDataCount, std::plus<int>(), root);//get all data counts
                std::vector<int> gatherSerializedData(globalDataCount);
                boost::mpi::gatherv(world, serializedData, gatherSerializedData.data(), gatherDataCountList, root);
                
                //Deserialize plain data into complex data
                outData.resize(globalItemCount);
                int _pointer = 0;
                for(int i = 0; i < globalItemCount; i++)
                {
                    int _data_count = gatherItemDataCountList.at(i);
                    
                    std::pair<int, int> _image_pairs;
                    _image_pairs.first() = gatherSerializedData.at(_pointer + 0);
                    _image_pairs.second() = gatherSerializedData.at(_pointer + 1);
                    
                    std::vector<std::pair<int, int>> _feature_pairs;
                    for(int j = 2; j < _data_count; j+=2)
                    {
                        _feature_pairs.push_back(std::make_pair(gatherSerializedData.at(_pointer + j), gatherSerializedData.at(_pointer + j + 1)));
                    }
                    
                    _pointer = _pointer + _data_count;
                }
            }
        }
    }
}
