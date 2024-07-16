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



//Matching class for multi-process and multi-thread features
#include<limits>
#include<Eigen/Eigen>
#include<boost/mpi.hpp>
#include<boost/serialization/vector.hpp>


namespace YH3DR
{
    namespace Algorithm
    {

		class ImageFeatures{
			private:
				friend class boost::serialization::access;
				Eigen::MatrixXi imageFeatures;

				template<class Archive>
				void serialize(Archive& ar, const unsigned int version)
				{
					ar& imageFeatures;
				}

			public:
				ImageFeatures(Eigen::MatrixXi& inImageFeatures)
				{
					imageFeatures = inImageFeatures;
				}

				~ImageFeatures()
				{

				}

		};



		enum class YH3DR_FEATURE_MATCHING_TYPE{
			FEATURE_MATCHING_BRUTEFORCE,
			FEATURE_MATCHING_ANN,
			FEATURE_MATCHING_KDTREE
		};

        class UMatching
        {
            private: 

				std::pair<int, int> imagePairs;

				YH3DR_FEATURE_MATCHING_TYPE featureMatchingType = YH3DR_FEATURE_MATCHING_TYPE::FEATURE_MATCHING_BRUTEFORCE;//Default feature matching type
				std::vector<std::pair<int, int>> matchingFeatures;//Restore feature pairs
				
				double distThreshold = 0.7f; //Default distance threshold
				
            public:
				UMatching(YH3DR_FEATURE_MATCHING_TYPE inMatchingType, int inSrc, int inDst);
				UMatching(int inSrc, int inDst);
				~UMatching();
				
				void SetThreshold(double inThreshold);
				void SetMatchingType(YH3DR_FEATURE_MATCHING_TYPE inMatchingType);
				std::pair<int, int> GetImagePairs();
				YH3DR_FEATURE_MATCHING_TYPE GetMatchingType();
				double GetThreshold();
				
				std::vector<std::pair<int, int>> GetMatchingFeatures();
				void GetMatchingFeaturesByBruteForce(Eigen::MatrixXi& d0, Eigen::MatrixXi& d1);
				void GetMatchingFeaturesByFLANN(Eigen::MatrixXi& d0, Eigen::MatrixXi& d1);
				void GetMatchingFeaturesByKDTree(Eigen::MatrixXi& d0, Eigen::MatrixXi& d1);

        };

    }
}
