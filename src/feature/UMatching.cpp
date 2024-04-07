
#include<iostream>
#include "UMatching.h"

namespace XSFM
{
    namespace Algorithm
    {
		UMatching::UMatching(XSFM_IMAGE_MATCHING_TYPE inMatchingType, const boost::mpi::environment* pInEnv, const boost::mpi::communicator* pInWorld)
		{
			pEnv = pInEnv;
			pWorld = pInWorld;

			//Check descriptor matcher type
			matchingType = inMatchingType;



			//Create descriptor matcher with matcher type

		}

		UMatching::~UMatching()
		{
		}



		std::vector<std::pair<int, int>> UMatching::GetMatchingFeaturesByBruteForce(Eigen::MatrixXd& d0, Eigen::MatrixXd& d1)
		{

			//Compute feature distances for every two images
			int _d0_size = d0.cols();//column major
			int _d1_size = d1.cols();//column major

			std::vector<int> _best_matchings_d0(_d0_size);
			std::vector<int> _best_matchings_d1(_d1_size);
			
			#pragma omp parallel for
			for(int i = 0; i < _d0_size; i++)
			{
				double minDistance = std::numeric_limits<double>::infinity();
				int minIndex = -1;
				for(int j = 0; j < _d1_size; j++)
				{
					Eigen::VectorXd error = d0.col(i) - d1.col(j);
					double _dist = error.norm();

					if(_dist < minDistance)
					{
						minDistance = _dist;
					}
				}

				if(minIndex > -1 && minDistance <= distThreshold)
				{
					_best_matchings_d0.at(i) = minIndex;
				}
				else
				{
					_best_matchings_d0.at(i) = -1;
				}
			}

			#pragma omp parallel for
			for(int i = 0; i < _d1_size; i++)
			{
				double minDistance = std::numeric_limits<double>::infinity();
				int minIndex = -1;

				for(int j = 0; j < _d0_size; j++)
				{
					Eigen::VectorXd error = d1.col(i) - d0.col(j);
					double _dist = error.norm();

					if(_dist < minDistance)
					{
						minDistance = _dist;
					}
				}

				if(minIndex > -1 && minDistance <= distThreshold)
				{
					_best_matchings_d1.at(i) = minIndex;
				}
				else
				{
					_best_matchings_d1.at(i) = -1;
				}
			}

			//Check valid feature matchings 
			std::vector<std::pair<int, int>> bestMatchings;
			for(int i = 0; i < _d0_size; i++)
			{
				if(_best_matchings_d0.at(i) == -1)
				{
					continue;
				}

				int _matching = _best_matchings_d0.at(i);
				if(_best_matchings_d1.at(_matching) == i)
				{
					bestMatchings.push_back(std::make_pair(i, _matching));
				}
			}
			
			return bestMatchings;
		}

		std::vector<std::pair<int, int>> GetMatchingFeaturesByFLANN(Eigen::MatrixXd& d0, Eigen::MatrixXd& d1)
		{

		}

		std::vector<std::pair<int, int>> GetMatchingFeaturesByKDTree(Eigen::MatrixXd& d0, Eigen::MatrixXd& d1)
		{

		}



		void UMatching::MatchingImages(const std::vector<Eigen::MatrixXd>& localImageFeatures)
		{
			switch(matchingType)
			{
				case XSFM_IMAGE_MATCHING_TYPE::IMAGE_MATCHING_BRUTEFORCE:
					MatchingImagesByBruteForce(localImageFeatures);
					break;
					
				case XSFM_IMAGE_MATCHING_TYPE::IMAGE_MATCHING_SEQUENCE:
					MatchingImagesBySequence(localImageFeatures);
					break;

				case XSFM_IMAGE_MATCHING_TYPE::IMAGE_MATCHING_VOCABULARY_TREE:
					MatchingImagesByVocabularyTree(localImageFeatures);
					break;

				default:
					break;
			}

		}


		//Create image pairs for every two images
		void UMatching::MatchingImagesByBruteForce(const std::vector<Eigen::MatrixXd>& localImageFeatures)
		{
			//Collect all image features for new distributions
			std::vector<Eigen::MatrixXd> allImageFeatures;
			int _count = localImageFeatures.size();
			std::vector<int> gatherImageCountList(pWorld->size());
			boost::mpi::all_gather(*pWorld, _count, gatherImageCountList);

			int allImageCount = 0;
			for(int i = 0; i < gatherImageCountList.size(); i++)
			{
				allImageCount += gatherImageCountList.at(i);

			}


			XSFM_PRINT("[Log_Level0_Info] Begin serialization for complex data.");
			//allImageFeatures.resize(_collect);

			//To be optimized
			//Unpack COMPLEX data for mpi transmission
			int localDataCount = 0;//Get total feature count for local images
			std::vector<int> featureCountList;//Get data structures
			for(int i = 0; i < localImageFeatures.size(); i++)
			{
				int featureCount = localImageFeatures.at(i).cols();
				int featureDim = localImageFeatures.at(i).rows();
				localDataCount = localDataCount + featureDim * featureCount;


				featureCountList.push_back(featureCount);//Record feature dimensions
			}

			//Gather the individual image feature count
			std::vector<int> gatherFeatureCountList(allImageCount);
			boost::mpi::gatherv(*pWorld, featureCountList, gatherFeatureCountList.data(), gatherImageCountList, 0);
			boost::mpi::broadcast(*pWorld, gatherFeatureCountList.data(), gatherFeatureCountList.size(), 0);



			//Gather the count of all image features
			std::vector<int> gatherDataCountList;
			boost::mpi::all_gather(*pWorld, localDataCount, gatherDataCountList);

			int allDataCount = 0;
			for(int i = 0; i < gatherDataCountList.size(); i++)
			{
				allDataCount += gatherDataCountList.at(i);
			}
			
			
			//Fill serialized data
			std::vector<double> serializedLocalImageFeatures(localDataCount);
			int index = 0;
			for(int i = 0; i < localImageFeatures.size(); i++)
			{
				int featureCount = localImageFeatures.at(i).cols();
				int featureDim = localImageFeatures.at(i).rows();
				for(int j = 0; j < featureCount; j++)
				{
					for(int k = 0; k < featureDim; k++)
					{

						serializedLocalImageFeatures.at(index++) = localImageFeatures.at(i)(k, j);
					}
				}
			}

			//Gather serialized image features with different lengths to root process
			std::vector<double> serializedAllImageFeatures(allDataCount);
			boost::mpi::gatherv(*pWorld, serializedLocalImageFeatures, serializedAllImageFeatures.data(), gatherDataCountList, 0);
	
			//Send collected image features to non-root processes
			boost::mpi::broadcast(*pWorld, serializedAllImageFeatures.data(), allDataCount, 0);
			XSFM_PRINT("[Log_Level0_Info] Data serialization finished...");


			//Unserialize image feature data *************************************************************************************
			std::vector<Eigen::MatrixXd> deserializedAllImageFeatures(allImageCount);
			int offset = 0;
			for(int i = 0; i < allImageCount; i++)
			{
				int row = featureDim;
				int col = gatherFeatureCountList.at(i);

				
				deserializedAllImageFeatures.at(i).resize(row, col);
				for(int j = 0; j < col; j++)
				{

					deserializedAllImageFeatures.at(i).col(j) = Eigen::Map<Eigen::VectorXd>(serializedLocalImageFeatures.data() + offset, row);
				}

				offset = offset + row;
			}


			std::cout<<deserializedAllImageFeatures.size()<<", "<<deserializedAllImageFeatures.at(0).cols()<<", "<<deserializedAllImageFeatures.at(0).rows()<<std::endl;
			//Load equilibrium for task distribution


		

		}


		void UMatching::MatchingImagesBySequence(const std::vector<Eigen::MatrixXd>& localImageFeatures)
		{

		}


		void UMatching::MatchingImagesByVocabularyTree(const std::vector<Eigen::MatrixXd>& localImageFeatures)
		{
		}
    }
}
