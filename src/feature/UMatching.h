

//Matching class for multi-process and multi-thread features
#include<limits>
#include<Eigen/Eigen>
#include<boost/mpi.hpp>
#include<boost/serialization/vector.hpp>

#define XSFM_PRINT(message) if(pWorld->rank() == 0) {std::cout<<message<<std::endl;}

//boost::mpi::environment env;
//boost::mpi::environment world;

namespace XSFM
{
    namespace Algorithm
    {

		class ImageFeatures{
			private:
				friend class boost::serialization::access;
				Eigen::MatrixXd imageFeatures;

				template<class Archive>
				void serialize(Archive& ar, const unsigned int version)
				{
					ar& imageFeatures;
				}

			public:
				ImageFeatures(Eigen::MatrixXd& inImageFeatures)
				{
					imageFeatures = inImageFeatures;
				}

				~ImageFeatures()
				{

				}

		};

		enum class XSFM_IMAGE_MATCHING_TYPE{
			IMAGE_MATCHING_BRUTEFORCE,
			IMAGE_MATCHING_SEQUENCE,
			IMAGE_MATCHING_VOCABULARY_TREE
		};


		enum class XSFM_FEATURE_MATCHING_TYPE{
			FEATURE_MATCHING_BRUTEFORCE,
			FEATURE_MATCHING_ANN,
			FEATURE_MATCHING_KDTREE
		};

        class UMatching
        {
            private: 
				boost::mpi::environment* pEnv;
				boost::mpi::communicator* pWorld;

				XSFM_IMAGE_MATCHING_TYPE matchingType;
				XSFM_FEATURE_MATCHING_TYPE featureMatchingType;
				std::pair<int, int> matchingPairs;
				
				double distThreshold = 0.7f;
				int featureDim = 128;
				std::vector<std::pair<int, int>> GetMatchingFeaturesByBruteForce(Eigen::MatrixXd& d0, Eigen::MatrixXd& d1);
				std::vector<std::pair<int, int>> GetMatchingFeaturesByFLANN(Eigen::MatrixXd& d0, Eigen::MatrixXd& d1);
				std::vector<std::pair<int, int>> GetMatchingFeaturesByKDTree(Eigen::MatrixXd& d0, Eigen::MatrixXd& d1);


            public:
				UMatching(XSFM_IMAGE_MATCHING_TYPE inMatchingType, const boost::mpi::environment* pInEnv, const boost::mpi::communicator* pInWorld);
				~UMatching();
				void MatchingImages(const std::vector<Eigen::MatrixXd>& localImageFeatures);
				void MatchingImagesByBruteForce(const std::vector<Eigen::MatrixXd>& localImageFeatures);
				void MatchingImagesBySequence(const std::vector<Eigen::MatrixXd>& localImageFeatures);
				void MatchingImagesByVocabularyTree(const std::vector<Eigen::MatrixXd>& localImageFeatures);

        };

    }
}
