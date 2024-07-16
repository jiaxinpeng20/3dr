//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the names of its 
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)


#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <filesystem>
#include <omp.h>
#include <chrono>
#include <thread>

#include "xsfm.h"
#include "feature/UImage.h"
#include "feature/UMatching.h"
//#include "feature/UImageMatching.h"

boost::mpi::environment env;
boost::mpi::communicator world;

#define PRINT_INTERVAL 5 //print every 5 images

double GetTimePointFromStart( std::chrono::steady_clock::time_point inOriginalClock)
{
    std::chrono::steady_clock::time_point elapsedClock = std::chrono::steady_clock::now();
	std::chrono::duration<double> duration = elapsedClock - inOriginalClock;
	return std::chrono::duration<double>(duration).count();
}

void logger();


//Sort images by image names
/*    std::sort(imagePathList.begin(), imagePathList.end(), [](std::string a, std::string b)
                                                            {
                                                                std::filesytem::path _path_a(a);
                                                                std::filesytem::path _path_b(b);
                                                                std::string _image_name_a = _path_a.filename().string();
                                                                std::string _image_name_b = _path_b.filename().string();
                                                                
                                                                return _image_name_a.compare(_image_name_b) > 0;
                                                            });*/
                                                            
                                                            
                                                            

std::vector<std::string> GetImagePathList(std::string imagePath, int& imageIndexOffset)
{
    std::vector<std::string> imagePathList(0), fullImagePathList(0);
    
    //Check the validity of the input image path
    if(!std::filesystem::exists(imagePath))
    {
       std::cout<<"Invalid image path..."<<std::endl;
       return imagePathList;
    }   
     
    int s = 0;
    for(auto iter = std::filesystem::directory_iterator(imagePath); iter != std::filesystem::directory_iterator(); iter++) 
    {
        if (std::filesystem::is_regular_file(*iter)) 
        {
            const std::filesystem::path file_path = *iter;
            fullImagePathList.push_back(file_path.string());
        }
        
        s++;
        if(s >= 128) break;
    }

    int resSize =  fullImagePathList.size() % world.size();
	int blockSize = fullImagePathList.size() / world.size();
    std::vector<int> arrangement(world.size() + 1, blockSize);
    arrangement.at(0) = 0;
    for(int i = 0; i < resSize; i++)
    {
        int _index = world.size() -1 - i;
        arrangement.at(_index)++;
    }

    for(int i = 0; i < arrangement.size()-1; i++)
    {
        arrangement.at(i+1) = arrangement.at(i+1) + arrangement.at(i);
    }

  
    //e.g. 22 images are distributed into 5 processes: {4, 4, 4, 5, 5}
    imageIndexOffset = arrangement.at(world.rank());
	for(int i = 0; i < fullImagePathList.size(); i++)//Splitting all images into multiple processes
	{
	    if(i >= arrangement.at(world.rank()) && i < arrangement.at(world.rank()+1))
	    {
	        imagePathList.push_back(fullImagePathList.at(i));
	    }
	}

  std::cout<<"###"<<imagePathList.size()<<std::endl;
    
    
    return imagePathList;

}


int main(int argc, char** argv)
{

    boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
    MPI_PRINT("******************************************YH3DR Projects******************************************");
    MPI_PRINT("                                                   ----3D Reconstruction software package by NUDT.");
    MPI_PRINT("");
    MPI_PRINT("Author: Jiaxin Peng");
   
    //Set multiple threads for local SMP node

    std::chrono::steady_clock::time_point originalClock = std::chrono::steady_clock::now();
    int maxThreads = std::thread::hardware_concurrency() + 1;
    int inNumOfThreads = -1;  //Reserved for manual settings
    if(inNumOfThreads <= 0 || inNumOfThreads > maxThreads)
    {
        inNumOfThreads = maxThreads;
    }
    int numOfThreads = inNumOfThreads;
    
    //Create local image lists and image pairs
    std::vector<YH3DR::Algorithm::UImage> imageList;
    std::vector<std::pair<int, int>> imagePairs;
    
    std::string imagePath(argv[1]);
    int imageIndexOffset;
	std::vector<std::string> imagePathList = GetImagePathList(imagePath, imageIndexOffset);
	imageList.resize(imagePathList.size());
	
	if(world.rank() == 0)
	{
	    double tt = GetTimePointFromStart(originalClock);
	    std::cout<<std::setprecision(8)<<"["<<tt<<"]"<<"Read images from file..."<<std::endl;
	}
	
	boost::mpi::timer readImageTimer;
	#pragma omp parallel for num_threads(numOfThreads)
	for(int i = 0; i < imageList.size(); i++)
	{
	    std::string imagePath = imagePathList.at(i);
	    imageList.at(i) = new YH3DR::Algorithm::UImage(imagePath);
	    std::filesystem::path _path;
	    imageList.at(i).SetImageID(i + imageIndexOffset);
	    imageList.at(i).SetImagePath(imagePath);
	    
	    imageList.at(i).SetFeatureDim(128);
	    imageList.at(i).SetFeatureCount(8192);
	    imageList.at(i).ReadImageFromPath();
	    imageList.at(i).GetImageExif();
	}
	double readImageTime = readImageTimer.elapsed();
	world.barrier();
	if(world.rank() == 0)
	{
	    double tt = GetTimePointFromStart(originalClock);
	    std::cout<<std::setprecision(8)<<"["<<tt<<"]"<<"Reading images is finished, which costs "<<readImageTime<<" seconds. "<<std::endl;
	}

  
	//Make sure that openCV is already multi-threaded.
	MPI_PRINT("Extract image features ...");
	boost::mpi::timer featureExtractionTimer;
	for(int i = 0; i < imageList.size(); i++)
	{
	    imageList.at(i).ComputeImageFeatures(); 
	     
	    if(world.rank() == 0)
	    {
	        //int bulk = imageList.size() > PRINT_COUNT ? PRINT_COUNT : imageList.size();
	        if(i % PRINT_INTERVAL == 0)
	        {    
	            double tt = GetTimePointFromStart(originalClock);   
	            std::cout<<std::setprecision(8)<<"["<<tt<<"]"<<"Feature extraction progress: " << (double)i*100/imageList.size() <<"%."<<std::endl;
	        }
	    } 
	}
	double featureExtractionTime = featureExtractionTimer.elapsed();
	world.barrier();
	if(world.rank() == 0)
	{
	    double tt = GetTimePointFromStart(originalClock);
	    std::cout<<std::setprecision(8)<<"["<<tt<<"]"<<"Feature extraction is finished, which costs "<<featureExtractionTime<<" seconds. "<<std::endl;
	}
	
  
 
    //Gather images from all nodes
    int localImageCount = imageList.size();
    std::vector<int> gatherImageCountList;
    boost::mpi::gather(world, localImageCount, gatherImageCountList, 0);
    
    
    
    
    //YH3DR::Algorithm::UImage localImage = imageList.at(0);
    //boost::mpi::gather(world, localImage, 2);
    //std::cout<<"#"<<world.rank() <<"=> "<< localImage.size()<<std::endl;
    
    //std::vector<YH3DR::Algorithm::UPackedImage> gatherPackedImage;
    //boost::mpi::gather(world, packedImage, gatherPackedImage, 0);
    
    
    
    MPI_PRINT("Compare image features...");
    boost::mpi::timer matchImageTimer;
    std::vector<YH3DR::Algorithm::UMatching> matchingList;
    for(int p = 0; p < world.size(); p++)
    {
        YH3DR::Algorithm::UPackedImage packedImage;
        
        if(world.rank() == p)//sending
        {
            packedImage.SetData(imageList);
            boost::mpi::broadcast(world, packedImage, p);
            std::cout<<p<<std::endl;
            
            //Diagonal blocks
            #pragma omp parallel for
            for(int i = 0; i < imageList.size() - 1; i++)
            {
                for(int j = i + 1; j < imageList.size(); j++)
                {
                    YH3DR::Algorithm::UMatching matching(imageList.at(i).GetImageID(), imageList.at(j).GetImageID());
                  
                    Eigen::MatrixXi _d0 = imageList.at(i).GetFeatureDescriptor();
                    Eigen::MatrixXi _d1 = imageList.at(j).GetFeatureDescriptor();
                    matching.GetMatchingFeaturesByFLANN(_d0, _d1);
                    
                    #pragma omp critical
                    matchingList.push_back(matching);
                }
            }
            
        }
        else //receiving
        {
            boost::mpi::broadcast(world, packedImage, p);
            
            //Compare every two images
            std::vector<YH3DR::Algorithm::UImage> refImageList;
            refImageList = packedImage.Unpack();//Unpack packedImage
            
            
            //Get the whole set of image pairs
            std::vector<std::pair<int, int>> _image_pairs;
            for(int i = 0; i < imageList.size(); i++)
            {
                for(int j = 0; j < refImageList.size(); j++)
                {
                    _image_pairs.push_back(std::make_pair(i, j));
                }
            }
            int _image_pair_count = _image_pairs.size();
            
            std::cout<<"#"<<world.rank()<<", "<<_image_pair_count<<std::endl;
            
            if(world.rank() > p)
            {
                #pragma omp parallel for
                for(int i = 0; i < _image_pair_count/2; i++)
                {
                    int _src = _image_pairs.at(i).first;
                    int _dst = _image_pairs.at(i).second;

                    YH3DR::Algorithm::UImage srcImage = imageList.at(_src);
                    YH3DR::Algorithm::UImage dstImage = refImageList.at(_dst);
                    YH3DR::Algorithm::UMatching matching(srcImage.GetImageID(), dstImage.GetImageID());
                    
                    Eigen::MatrixXi _d0 = srcImage.GetFeatureDescriptor();
                    Eigen::MatrixXi _d1 = dstImage.GetFeatureDescriptor();
                    matching.GetMatchingFeaturesByFLANN(_d0, _d1);
                    #pragma omp critical
                    matchingList.push_back(matching);
                }
             }
             else if(world.rank() < p)
             {
                #pragma omp parallel for
                for(int i = _image_pair_count/2; i < _image_pair_count; i++)
                {
                    int _src = _image_pairs.at(i).first;
                    int _dst = _image_pairs.at(i).second;

                    YH3DR::Algorithm::UImage srcImage = imageList.at(_src);
                    YH3DR::Algorithm::UImage dstImage = refImageList.at(_dst);
                    YH3DR::Algorithm::UMatching matching(srcImage.GetImageID(), dstImage.GetImageID());
                    
                    Eigen::MatrixXi _d0 = srcImage.GetFeatureDescriptor();
                    Eigen::MatrixXi _d1 = dstImage.GetFeatureDescriptor();
                    matching.GetMatchingFeaturesByFLANN(_d0, _d1);
                    #pragma omp critical
                    matchingList.push_back(matching);
                }
             }
            
        }  
        
    }
    double matchImageTime = matchImageTimer.elapsed();
    world.barrier();
    if(world.rank() == 0)
	{
	    double tt = GetTimePointFromStart(originalClock);
	    std::cout<<std::setprecision(8)<<"["<<tt<<"]"<<"Matching images is finished, which costs "<<matchImageTime<<" seconds. "<<std::endl;
	}
    
    
    
	//Gather matchings
	

 	

    
    //XSFM::Algorithm::UMatching matching(XSFM::Algorithm::XSFM_IMAGE_MATCHING_TYPE::IMAGE_MATCHING_BRUTEFORCE, imageIndexOffset, &env, &world);
    //matching.MatchingImages(descriptors);


     

	
	//RunFeatureExtractor(imageList);
	
	
	//std::cout<<world.rank()<<", "<<world.size()<<", "<<imageList.size()<<std::endl;

}
