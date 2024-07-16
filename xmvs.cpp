
#include <boost/mpi.hpp>
#include <boost/filesystem.hpp>

#include "xsfm.h"
#include "UImage.h"
#include "feature/UMatching.h"

boost::mpi::environment env;
boost::mpi::communicator world;

//typedef MPI_PRINT()


std::vector<std::string> GetImageList(std::string imagePath, int& imageIndexOffset)
{
    std::vector<std::string> imageList(0), fullImageList(0);
    
    //Check the validity of the input image path
    if(!boost::filesystem::exists(imagePath))
    {
       std::cout<<"Invalid image path..."<<std::endl;
       return imageList;
    }   
     
    
    for(auto iter = boost::filesystem::directory_iterator(imagePath); iter != boost::filesystem::directory_iterator(); iter++) 
    {
        if (boost::filesystem::is_regular_file(*iter)) 
        {
            const boost::filesystem::path file_path = *iter;
            fullImageList.push_back(file_path.string());
        }
    }

  int resSize =  fullImageList.size() % world.size();
	int blockSize = fullImageList.size() / world.size();
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
	for(int i = 0; i < fullImageList.size(); i++)//Splitting all images into multiple processes
	{
	    if(i >= arrangement.at(world.rank()) && i < arrangement.at(world.rank()+1))
	    {
	        imageList.push_back(fullImageList.at(i));
	    }
	}

  std::cout<<"###"<<imageList.size()<<std::endl;
    
    
    return imageList;
    
}


int main(int argc, char** argv)
{
    boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
    DISTRIBUTE_PRINT("******************************************xSFM Projects******************************************");
    DISTRIBUTE_PRINT("                                                            ----Sparse 3D Reconstruction by NUDT.");
    DISTRIBUTE_PRINT("");
    DISTRIBUTE_PRINT("Author:");
    
	std::string imagePath(argv[1]);
	DISTRIBUTE_PRINT(std::string("Project Location:"+imagePath));
    int imageIndexOffset;
	std::vector<std::string> imageList = GetImageList(imagePath, imageIndexOffset);
	
    DISTRIBUTE_PRINT("Reading images from file...");
	XSFM::Algorithm::UImage image(imageList, &env, &world);
	image.ReadImageFromPath();
      

    DISTRIBUTE_PRINT("Compute image features...");
    image.ComputeImageFeatures();  
    std::vector<Eigen::MatrixXi> descriptors = image.GetImageDescriptors();

    DISTRIBUTE_PRINT("Compare image features...");
    XSFM::Algorithm::UMatching matching(XSFM::Algorithm::XSFM_IMAGE_MATCHING_TYPE::IMAGE_MATCHING_BRUTEFORCE, imageIndexOffset, &env, &world);
    matching.MatchingImages(descriptors);


     

	
	//RunFeatureExtractor(imageList);
	
	
	std::cout<<world.rank()<<", "<<world.size()<<", "<<imageList.size()<<std::endl;
	
}
