

#include "UImageMatching.h"



namespace YH3DR
{
    namespace Algorithm
    {
        UImageMatching::UImageMatching()
        {
            
        }
        
        UImageMatching::~UImageMatching()
        {
            
        }
        
        
        std::vector<std::pair<int, int>> MatchingImagesByBruteForce(std::vector<YH3DR::Algorithm::UImage>& inImageList)
        {
            std::vector<std::pair<int, int>> matchingPairs;
            
            int imageCount = inImageList.size();
            for(int i = 0; i < imageCount-1; i++)
            {
                int src = inImageList.at(i).GetImageID();
                for(int j = i + 1; j < imageCount; j++)
                {
                    int dst = inImageList.at(j).GetImageID();
                            
                    if(src > 0 && dst > 0 && src != dst)
                    {
                        matchingPairs.push_back(std::make_pair(src, dst));
                    }
                }
            }
            
            
            return matchingPairs;
        }
        
        std::vector<std::pair<int, int>> MatchingImagesBySequence(std::vector<YH3DR::Algorithm::UImage>& inImageList)
        {
            std::vector<std::pair<int, int>> matchingPairs;
            return matchingPairs;
        }
        
        std::vector<std::pair<int, int>> MatchingImagesByVocabularyTree(std::vector<YH3DR::Algorithm::UImage>& inImageList)
        {
            std::vector<std::pair<int, int>> matchingPairs;
            return matchingPairs;
        }
        
        std::vector<std::pair<int, int>> UImageMatching::GetMatchingPairs(std::vector<YH3DR::Algorithm::UImage>& inImageList, YH3DR::Algorithm::MatchingType inMatchingType)
        {
            
            
            switch(inMatchingType)
            {
                
                case MatchingType::IMAGE_MATCHING_BRUTEFORCE:
                    return MatchingImagesByBruteForce(inImageList);   
                break;
                
                case MatchingType::IMAGE_MATCHING_SEQUENCE:
                    return MatchingImagesBySequence(inImageList);
                break;
                
                case MatchingType::IMAGE_MATCHING_VOCABULARY_TREE:
                    return MatchingImagesByVocabularyTree(inImageList);
                break;
                
                default:
                    return std::vector<std::pair<int, int>>(0);
                break;
            }
            
        }
    }
}
