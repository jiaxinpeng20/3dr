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


#ifndef YH3DR_LIB3DR_SERIALIZE_H
#define YH3DR_LIB3DR_SERIALIZE_H


#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#define WITH_BOOST_SERIALIZATION_SPLIT_FREE 1

namespace boost
{
    namespace serialization
    {

        //Serialization for cv::Mat
#ifndef WITH_BOOST_SERIALIZATION_SPLIT_FREE
        template<class Archive>
        void serialize(Archive& ar, cv::Mat& mat, const unsigned int version)
        {
            int rows, cols, type;
            bool continuous;
            
            if(Archive::is_saving::value)
            {
                rows = mat.rows;
                cols = mat.cols;
                type = mat.type();
                continuous = mat.isContinuous();
            }
            
            ar& BOOST_SERIALIZATION_NVP(rows);
            ar& BOOST_SERIALIZATION_NVP(cols);
            ar& BOOST_SERIALIZATION_NVP(type);
            ar& BOOST_SERIALIZATION_NVP(continuous);
            
            if(Archive::is_loading::value)
            {
                mat.create(rows, cols, type);
            }
            
            if(continuous)
            {
                const int data_size = rows * cols * mat.elemSize();
                ar& boost::serialization::make_array(mat.ptr(), data_size);
            }
            else
            {
                const int row_size = cols * mat.elemSize();
                for (int i = 0; i < rows; i++) 
                {
                    ar& boost::serialization::make_array(mat.ptr(i), row_size);
                }
            }
        }
#else
        template<class Archive>
        void save(Archive& ar, const cv::Mat& mat, const unsigned int version)
        {
            int rows = mat.rows;
            int cols = mat.cols;
            int type = mat.type();
            bool continuous = mat.isContinuous();
            
            ar& BOOST_SERIALIZATION_NVP(rows);
            ar& BOOST_SERIALIZATION_NVP(cols);
            ar& BOOST_SERIALIZATION_NVP(type);
            ar& BOOST_SERIALIZATION_NVP(continuous);
            
            if(continuous)
            {
                const int data_size = rows * cols * mat.elemSize();
                ar& boost::serialization::make_array(mat.ptr(), data_size);
            }
            else
            {
                const int row_size = cols * mat.elemSize();
                for (int i = 0; i < rows; i++) 
                {
                    ar& boost::serialization::make_array(mat.ptr(i), row_size);
                }
            }
        }
        
        template<class Archive>
        void load(Archive& ar, cv::Mat& mat, const unsigned int version)
        {
            int rows, cols, type;
            bool continuous;
            
            ar& BOOST_SERIALIZATION_NVP(rows);
            ar& BOOST_SERIALIZATION_NVP(cols);
            ar& BOOST_SERIALIZATION_NVP(type);
            ar& BOOST_SERIALIZATION_NVP(continuous);
            
            
            mat.create(rows, cols, type);
            
            if(continuous)
            {
                const int data_size = rows * cols * mat.elemSize();
                ar& boost::serialization::make_array(mat.ptr(), data_size);
            }
            else
            {
                const int row_size = cols * mat.elemSize();
                for (int i = 0; i < rows; i++) 
                {
                    ar& boost::serialization::make_array(mat.ptr(i), row_size);
                }
            }
        }
        
        template<class Archive>
        void serialize(Archive& ar, cv::Mat& mat, const unsigned int version)
        {
            split_free(ar, mat, version);
        }
        
#endif


        //Serialization for Eigen::Matrix with split_free mode
        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        void save(Archive& ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat, const unsigned int version)
        {
            int rows = mat.rows();
            int cols = mat.cols();
            ar& rows;
            ar& cols;
            ar& boost::serialization::make_array(mat.data(), rows * cols);
            
        }
        
        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        void load(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat, const unsigned int version)
        {
            int rows, cols;
            ar& rows;
            ar& cols;
            mat.resize(rows, cols);
            ar& boost::serialization::make_array(mat.data(), rows * cols);
        }
        
        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        void serialize(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat, const unsigned int version)
        {
            split_free(ar, mat, version);
        }
        
        
        //Serialization for Eigen::SparseMatrix with split_free model
        template<class Archive, typename _Scalar, int _Options, typename _Index>
        void save(Archive& ar, Eigen::SparseMatrix<_Scalar, _Options, _Index>& sparseMat, const unsigned int version)
        {
            
        }
        
        template<class Archive, typename _Scalar, int _Options, typename _Index>
        void load(Archive& ar, Eigen::SparseMatrix<_Scalar, _Options, _Index>& sparseMat, const unsigned int version)
        {
            
        }
        
        template<class Archive, typename _Scalar, int _Options, typename _Index>
        void serialize(Archive& ar, Eigen::SparseMatrix<_Scalar, _Options, _Index>& sparseMat, const unsigned int version)
        {
            split_free(ar, sparseMat, version);
        }
    }
}

//Serialization by continuous buffer
//boost::mpi::packed_oarchive oa(world);
//oa << packedImage;
//auto bufferPtr = const_cast<void*>(oa.address());
//unsigned long bufferSize = static_cast<unsigned long>(oa.size());
//unsigned long gatherSize;
//boost::mpi::all_reduce(world, bufferSize, ga);
//boost::mpi::all_reduce(world, (unsigned long)bufferSize, gatherSize, std::plus<unsigned long>());
//std::cout<<bufferSize<<", "<<oa.size()<<", "<<gatherSize<<std::endl;
    
    
namespace YH3DR
{
    namespace Algorithm
    {
        
    }
}



#endif
