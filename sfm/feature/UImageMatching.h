
#include <vector>
#include <utility>
#include "feature/UImage.h"

namespace YH3DR
{
    namespace Algorithm
    {
        enum class MatchingType
        {
            IMAGE_MATCHING_BRUTEFORCE,
			IMAGE_MATCHING_SEQUENCE,
			IMAGE_MATCHING_VOCABULARY_TREE
        };
        
        class UImageMatching
        {
    
            private:
                static std::vector<std::pair<int, int>> MatchingImagesByBruteForce(std::vector<YH3DR::Algorithm::UImage>& inImageList);
                static std::vector<std::pair<int, int>> MatchingImagesBySequence(std::vector<YH3DR::Algorithm::UImage>& inImageList);
                static std::vector<std::pair<int, int>> MatchingImagesByVocabularyTree(std::vector<YH3DR::Algorithm::UImage>& inImageList);
                
            
            public:
                UImageMatching();
                ~UImageMatching();
                static std::vector<std::pair<int, int>> GetMatchingPairs(std::vector<YH3DR::Algorithm::UImage>& inImageList, YH3DR::Algorithm::MatchingType inMatchingType);
        };
        
    }
}
