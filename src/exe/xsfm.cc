

#include "exe/database.h"
#include "exe/feature.h"
#include "exe/image.h"
#include "exe/model.h"
#include "exe/sfm.h"
#include "exe/vocab_tree.h"
#include "util/version.h"
#include<boost/mpi.hpp>
#include <boost/filesystem.hpp>
#include "controllers/automatic_reconstruction.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "base/reconstruction.h"
#include <FreeImage.h>

#include "xsfm.h"
#include "base/UImage.h"
#include "feature/UMatching.h"

boost::mpi::environment env;
boost::mpi::communicator world;

//typedef MPI_PRINT()

using namespace colmap;

namespace {

typedef std::function<int(int, char**)> command_func_t;

int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  using namespace colmap;

  std::cout << StringPrintf(
                   "%s -- Structure-from-Motion and Multi-View Stereo\n"
                   "              (%s)",
                   GetVersionInfo().c_str(), GetBuildInfo().c_str())
            << std::endl
            << std::endl;

  std::cout << "Usage:" << std::endl;
  std::cout << "  colmap [command] [options]" << std::endl << std::endl;

  std::cout << "Documentation:" << std::endl;
  std::cout << "  https://colmap.github.io/" << std::endl << std::endl;

  std::cout << "Example usage:" << std::endl;
  std::cout << "  colmap help [ -h, --help ]" << std::endl;
  std::cout << "  colmap automatic_reconstructor -h [ --help ]" << std::endl;
  std::cout << "  colmap automatic_reconstructor --image_path IMAGES "
               "--workspace_path WORKSPACE"
            << std::endl;
  std::cout << "  colmap feature_extractor --image_path IMAGES --database_path "
               "DATABASE"
            << std::endl;
  std::cout << "  colmap exhaustive_matcher --database_path DATABASE"
            << std::endl;
  std::cout << "  colmap mapper --image_path IMAGES --database_path DATABASE "
               "--output_path MODEL"
            << std::endl;
  std::cout << "  ..." << std::endl << std::endl;

  std::cout << "Available commands:" << std::endl;
  std::cout << "  help" << std::endl;
  for (const auto& command : commands) {
    std::cout << "  " << command.first << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

}  // namespace


/*
int main(int argc, char** argv) {
  using namespace colmap;

  InitializeGlog(argv);
#ifdef GUI_ENABLED
  Q_INIT_RESOURCE(resources);
#endif

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("automatic_reconstructor", &RunAutomaticReconstructor);
  commands.emplace_back("exhaustive_matcher", &RunExhaustiveMatcher);
  commands.emplace_back("feature_extractor", &RunFeatureExtractor);
  commands.emplace_back("feature_importer", &RunFeatureImporter);
  commands.emplace_back("hierarchical_mapper", &RunHierarchicalMapper);
  

  if (argc == 1) {
    return ShowHelp(commands);
  }

  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr) {
      std::cerr << StringPrintf(
                       "ERROR: Command `%s` not recognized. To list the "
                       "available commands, run `colmap help`.",
                       command.c_str())
                << std::endl;
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }

}
*/

std::vector<std::string> GetImageList(std::string imagePath)
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
    
	
	for(int i = 0; i < fullImageList.size(); i++)//Splitting all images into multiple processes
	{
	    if(i%world.size() == world.rank())
	    {
	        imageList.push_back(fullImageList.at(i));
	    }
	}
    
    
    return imageList;
    
}


int RunFeatureExtractor(std::vector<std::string> imageList) {
  std::string image_list_path;
  int camera_mode = -1;
  std::string descriptor_normalization = "l1_root";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("camera_mode", &camera_mode);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("descriptor_normalization",
                           &descriptor_normalization, "{'l1_root', 'l2'}");
  options.AddExtractionOptions();
  //options.Parse(argc, argv);

  /*ImageReaderOptions reader_options = *options.image_reader;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  StringToLower(&descriptor_normalization);
  if (descriptor_normalization == "l1_root") {
    options.sift_extraction->normalization =
        SiftExtractionOptions::Normalization::L1_ROOT;
  } else if (descriptor_normalization == "l2") {
    options.sift_extraction->normalization =
        SiftExtractionOptions::Normalization::L2;
  } else {
    std::cerr << "ERROR: Invalid `descriptor_normalization`" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!ExistsCameraModelWithName(reader_options.camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
  }

  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
    return EXIT_FAILURE;
  }


  SiftFeatureExtractor feature_extractor(reader_options,
                                         *options.sift_extraction);

 
  feature_extractor.Start();
  feature_extractor.Wait();*/

  return EXIT_SUCCESS;
}


int main(int argc, char** argv)
{
  boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
	
  std::cout<<"MPI"<<std::endl;
	std::string imgPath(argv[1]);
	std::vector<std::string> imageList = GetImageList(imgPath);
	
  DISTRIBUTE_PRINT("Reading images from file...");
	XSFM::Algorithm::UImage image(imageList);
	image.ReadImageFromPath();
      

  DISTRIBUTE_PRINT("Compute image features...");
  image.ComputeImageFeatures();  
  std::vector<Eigen::MatrixXd> descriptors = image.GetImageDescriptors();

  DISTRIBUTE_PRINT("Compare image features...");
  for(int i = 0; i < imageList.size() - 1; i++)
  {
    for(int j = i+1; j < imageList.size(); j++)
    {
      DISTRIBUTE_PRINT("Compare feature distances for every two images...");
      Eigen::MatrixXd _d0 = descriptors.at(i);
      Eigen::MatrixXd _d1 = descriptors.at(j);
      XSFM::Algorithm::UMatching matching(0, &env, &world);
      matching.MatchingFeaturesByBruteForce(_d0, _d1);
    }
  }

     

	
	//RunFeatureExtractor(imageList);
	
	
	std::cout<<world.rank()<<", "<<world.size()<<", "<<imageList.size()<<std::endl;
	
}
