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


#include<iostream>
#include "UMatching.h"
#include "flann/flann.hpp"

namespace YH3DR
{
    namespace Algorithm
    {
		UMatching::UMatching(YH3DR_FEATURE_MATCHING_TYPE inMatchingType, int inSrc, int inDst)
		{

			//Check descriptor matcher type
			this->featureMatchingType = inMatchingType;
			this->imagePairs = std::make_pair(inSrc, inDst);
			this->matchingFeatures.resize(0);
		}
		
		UMatching::UMatching(int inSrc, int inDst)
		{
		    this->imagePairs = std::make_pair(inSrc, inDst);
		    this->matchingFeatures.resize(0);
		}

		UMatching::~UMatching()
		{
		}



        void UMatching::SetThreshold(double inThreshold)
        {
            this->distThreshold = inThreshold;
        }
        
        void UMatching::SetMatchingType(YH3DR_FEATURE_MATCHING_TYPE inMatchingType)
        {
            this->featureMatchingType = inMatchingType;
        }
        
        std::pair<int, int> UMatching::GetImagePairs()
        {
            return this->imagePairs;
        }
        
        YH3DR_FEATURE_MATCHING_TYPE UMatching::GetMatchingType()
        {
            return this->featureMatchingType;
        }
        
        double UMatching::GetThreshold()
        {
            return this->distThreshold;
        }
        
        std::vector<std::pair<int, int>> UMatching::GetMatchingFeatures()
        {
            return this->matchingFeatures;
        }



		void UMatching::GetMatchingFeaturesByBruteForce(Eigen::MatrixXi& d0, Eigen::MatrixXi& d1)
		{

			//Compute feature distances for every two images
			int _d0_size = d0.cols();//column major
			int _d1_size = d1.cols();//column major

			std::vector<int> _best_matchings_d0(_d0_size);
			std::vector<int> _best_matchings_d1(_d1_size);
			
			#pragma omp parallel for
			for(int i = 0; i < _d0_size; i++)
			{
				double minDistance = std::numeric_limits<int>::infinity();
				int minIndex = -1;
				for(int j = 0; j < _d1_size; j++)
				{
					Eigen::VectorXi error = d0.col(i) - d1.col(j);
					int _dist = error.norm();

					if(_dist < minDistance)
					{
						minIndex = j;
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
				int minDistance = std::numeric_limits<int>::infinity();
				int minIndex = -1;

				for(int j = 0; j < _d0_size; j++)
				{
					Eigen::VectorXi error = d1.col(i) - d0.col(j);
					int _dist = error.norm();

					if(_dist < minDistance)
					{
						minIndex = j;
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
			for(int i = 0; i < _d0_size; i++)
			{
				if(_best_matchings_d0.at(i) == -1)
				{
					continue;
				}

				int _matching = _best_matchings_d0.at(i);
				if(_best_matchings_d1.at(_matching) == i)
				{
					matchingFeatures.push_back(std::make_pair(i, _matching));
				}
			}
		
		}

		//Get Matching Features by Approximate Nearest Neighbor
		void UMatching::GetMatchingFeaturesByFLANN(Eigen::MatrixXi& d0, Eigen::MatrixXi& d1)
		{

			
			//Check EMPTY matrices
			if(d0.cols() == 0 || d1.cols() == 0)
			{
				return;
			}


			Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> indexMat;
			Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> distMat;

			auto CalculateKNN = [&indexMat, &distMat](Eigen::MatrixXi& d0, Eigen::MatrixXi& d1)
			{
				
				const int defaultNearestNeighbors = 2;
				int numOfFeatures = d0.cols();
				const int knn = std::min(defaultNearestNeighbors, numOfFeatures);

				
				indexMat.resize(numOfFeatures, knn);//row major
				distMat.resize(numOfFeatures, knn);//row major

				//Translate column-major matrices into row-major matrices since FLANN::Matrix is row major
				Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dt0(d0.rows(), d0.cols());
				Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dt1(d1.rows(), d1.cols());
				dt0 = d0;
				dt1 = d1;


				flann::Matrix<int> query(const_cast<int*>(dt0.data()), dt0.rows(), dt0.cols());
				flann::Matrix<int> database(const_cast<int*>(dt1.data()), dt1.rows(), dt1.cols());

				flann::Matrix<int> flannIndexMat(indexMat.data(), numOfFeatures, knn);
				std::vector<float> tempDistVec(numOfFeatures * knn);
				flann::Matrix<float> flannDistMat(tempDistVec.data(), numOfFeatures, knn);
			
				//Construct a randomized kd-tree with 4 kd-trees in the forest
				flann::Index<flann::L2<int>> index(database, flann::KDTreeIndexParams(4));
				index.buildIndex();
				index.knnSearch(query, flannIndexMat, flannDistMat, knn, flann::SearchParams(128));
				
				//Trancate tempDistVec to distMat
				for(int i = 0; i < numOfFeatures; i++)
				{
					for(int j = 0; j < knn; j++)
					{
						distMat.coeffRef(i, j) = tempDistVec.at(i*knn + j);
					}
				}

			};


			CalculateKNN(d0, d1);
			std::vector<int> _best_matchings_d0(d0.cols());
			#pragma omp parallel for
			for(int i = 0; i < indexMat.rows(); i++)
			{
				int minIndex = -1;
				int minDistance = std::numeric_limits<int>::infinity();

				for(int j = 0; j < indexMat.cols(); j++)
				{
					if(distMat(i, j) < minDistance)
					{
						minDistance = distMat(i, j);
						minIndex = indexMat(i, j);
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


			CalculateKNN(d1, d0);
			std::vector<int> _best_matchings_d1(d1.cols());
			#pragma omp parallel for
			for(int i = 0; i < indexMat.rows(); i++)
			{
				int minIndex = -1;
				float minDistance = std::numeric_limits<float>::infinity();

				for(int j = 0; j < indexMat.cols(); j++)
				{
					if(distMat(i, j) < minDistance)
					{
						minIndex = indexMat(i, j);
						minDistance = distMat(i, j);
					}
				}

				if(minIndex > -1 && minDistance <= distThreshold)
				{
					_best_matchings_d1.at(i) = minIndex;
				}
				else
				{
					_best_matchings_d1.at(i) = -1;
					//std::cout<<minIndex<<", "<<minDistance<<", "<<distThreshold<<std::endl;
				}
			}


			//Define matching pairs
			//Check valid feature matchings
			for(int i = 0; i < _best_matchings_d0.size(); i++)
			{
				int _matching = _best_matchings_d0.at(i);
				if(_matching == -1)
				{
					continue;
				}

				if(_best_matchings_d1.at(_matching) == i)
				{
					matchingFeatures.push_back(std::make_pair(i, _matching));
				}
			}

		}

		void UMatching::GetMatchingFeaturesByKDTree(Eigen::MatrixXi& d0, Eigen::MatrixXi& d1)
		{

		}

		

#if 0
		//Create image pairs for every two images
		void UMatching::MatchingImagesByBruteForce(std::vector<Eigen::MatrixXi>& localImageFeatures)
		{
	
			//Collect all image features for new distributions
			std::vector<Eigen::MatrixXi> allImageFeatures;
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
			std::vector<int> serializedLocalImageFeatures(localDataCount);
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
			std::vector<int> serializedAllImageFeatures(allDataCount);
			boost::mpi::gatherv(*pWorld, serializedLocalImageFeatures, serializedAllImageFeatures.data(), gatherDataCountList, 0);
	
			//Send collected image features to non-root processes
			boost::mpi::broadcast(*pWorld, serializedAllImageFeatures.data(), allDataCount, 0);
			XSFM_PRINT("[Log_Level0_Info] Data serialization finished...");


			//Deserialize image feature data *************************************************************************************
			XSFM_PRINT("[Log_Level0_Info] Deserialize feature data...");
			XSFM_PRINT("     Transform std::vector<Eigen::MatrixXd> data to serialized data.");
			std::vector<std::vector<Eigen::MatrixXi>> deserializedAllImageFeatures(pWorld->size());
			for(int i = 0, _image_index = 0, _feature_index = 0; i < pWorld->size(); i++)
			{
				int localImageCount = gatherImageCountList.at(i);
				deserializedAllImageFeatures.at(i).resize(localImageCount);


				for(int j = 0; j < localImageCount; j++)
				{
					int row = featureDim;
					int col = gatherFeatureCountList.at(_image_index++);
				
					deserializedAllImageFeatures.at(i).at(j).resize(row, col);
					for(int k = 0; k < col; k++)
					{
						deserializedAllImageFeatures.at(i).at(j).col(k) = Eigen::Map<Eigen::VectorXi>(serializedAllImageFeatures.data() + _feature_index, row);
						_feature_index = _feature_index + row;
					}

				}
			}
			XSFM_PRINT("[Log_Level0_Info] Data deserialization finished.");

			
			//Build matchings for task equilibrium
			int _s_offset = 0;
			for(int s = 0; s < pWorld->rank(); s++)
			{
				_s_offset = _s_offset + deserializedAllImageFeatures.at(s).size();
			}

			for(int i = 0; i < deserializedAllImageFeatures.size(); i++)
			{

				int _r_offset = 0;
				for(int s = 0; s < i; s++)
				{
					_r_offset += deserializedAllImageFeatures.at(s).size();
				}


				if(i == pWorld->rank())//Is diagonal block
				{
					#pragma omp  parallel for
					for(int j = 0; j < deserializedAllImageFeatures.at(i).size(); j++)//Lower triangles
					{
						for(int k = 0; k < j; k++)
						{
							//std::vector<std::pair<int, int>> _matching_pairs = GetMatchingFeaturesByBruteForce(deserializedAllImageFeatures.at(i).at(j), deserializedAllImageFeatures.at(i).at(k));
							std::vector<std::pair<int, int>> _matching_pairs = GetMatchingFeaturesByFLANN(deserializedAllImageFeatures.at(i).at(j), deserializedAllImageFeatures.at(i).at(k));
							
							#pragma omp critical
							matchingPairs.push_back(std::make_pair(std::make_pair(_r_offset + j, _r_offset + k), _matching_pairs));
						}
					}
				}



				if(i > pWorld->rank())//below the diagonal block
				{
					#pragma omp parallel for
					for(int j = 0; j < deserializedAllImageFeatures.at(i).size(); j++)
					{
						for(int k = 0; k <= j; k++)
						{
							//std::vector<std::pair<int, int>> _matching_pairs = GetMatchingFeaturesByBruteForce(localImageFeatures.at(k), deserializedAllImageFeatures.at(i).at(j));
							std::vector<std::pair<int, int>> _matching_pairs = GetMatchingFeaturesByFLANN(localImageFeatures.at(k), deserializedAllImageFeatures.at(i).at(j));
							
							#pragma omp critical
							matchingPairs.push_back(std::make_pair(std::make_pair(_r_offset + j, _s_offset + k), _matching_pairs));
						}

					}
				}

				if(i < pWorld->rank())//upon the diagonal block
				{
					#pragma omp parallel for
					for(int j = 0; j < localImageFeatures.size(); j++)
					{
						for(int k = j+1; k < deserializedAllImageFeatures.at(i).size(); k++)
						{
							//std::vector<std::pair<int, int>> _matching_pairs = GetMatchingFeaturesByBruteForce(localImageFeatures.at(j), deserializedAllImageFeatures.at(i).at(k));
							std::vector<std::pair<int, int>> _matching_pairs = GetMatchingFeaturesByFLANN(localImageFeatures.at(j), deserializedAllImageFeatures.at(i).at(k));
							
							#pragma omp critical
							matchingPairs.push_back(std::make_pair(std::make_pair(_s_offset + j, _r_offset + k), _matching_pairs));
						}
					}
				}
			}

			for(int k = 0; k < matchingPairs.size(); k++)
			{
				std::pair<int, int> _matching = matchingPairs.at(k).first;
				std::cout<<pWorld->rank()<<"  "<<"<"<<_matching.first<<", "<<_matching.second<<"> ";
			}
			std::cout<<std::endl;

			//Gather matching features from all processes
			std::vector<std::pair<std::pair<int, int>, std::vector<std::pair<int, int>>>> gatherMatchingPairs;
			XSFM::Communication::Gatherv(*pWorld, matchingPairs, gatherMatchingPairs, 0); 
			if(pWorld->rank() == 0)
			{
				std::cout<<"Matching Pairs: "<<gatherMatchingPairs.size()<<std::endl;
			}
		
			pWorld->barrier();

		}
#endif

    }
}
