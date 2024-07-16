//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
//
// Written by Dr. Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)


#include "fundamental_matrix.h"

namespace YH3DR
{
    namespace MultiviewGeometry
    {
        UFundamentalMatrix::UFundamentalMatrix()
        {
    
        }

        UFundamentalMatrix::~UFundamentalMatrix()
        {
    
        }

        std::vector<Eigen::Matrix3d> UFundamentalMatrix::SolveBy7PointMethod(Eigen::Matrix2d& p1, Eigen::Matrix2d& p2)
        {
            
            //Check the validity of input data
            if(p1.cols() != 7 || p2.cols() != 7)
            {
                return std::vector<Eigen::Matrix3d>(0);
            }
            
            
			//Build linear systems for Af = 0 from x'^TFx = 0
			Eigen::MatrixXd A(7, 9);//The coefficient matrix is 7x9
            A.col(0) = p1.row(0).cwiseProduct(p2.row(0)).transpose();
			A.col(1) = p1.row(0).cwiseProduct(p2.row(1)).transpose();
			A.col(2) = p1.row(0).transpose();
			A.col(3) = p1.row(1).cwiseProduct(p2.row(0)).transpose();
			A.col(4) = p1.row(1).cwiseProduct(p2.row(1)).transpose();
			A.col(5) = p1.row(1).transpose();
			A.col(6) = p2.row(0).transpose();
			A.col(7) = p2.row(1).transpose();
			A.col(8) = Eigen::VectorXd::Ones(7);

			//Solve linear systems with Jacobi SVD since Af = 0 is small
			//Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ColPivHouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeFullV> solver(A);
			Eigen::JacobiSVD<Eigen::MatrixXd> solver(A, Eigen::ComputeFullV);
			const Eigen::MatrixXd matV = solver.matrixV();
			Eigen::VectorXd f1 = matV.col(7);//Get right nullspaces, F1 is the 3x3 layout of f1;
			Eigen::VectorXd f2 = matV.col(8);//Get right nullspaces, F2 is the 3x3 layout of f2;

			//Satisfy the constraint det(\lambda F1 + (1-\lambda)F2) = 0
			Eigen::MatrixXd F1 = f1;
			Eigen::MatrixXd F2 = f2;
			F1.resize(3, 3);
			F2.resize(3, 3);
			double c00 = F1(0, 0) - F2(0, 0),\
				   c01 = F1(0, 1) - F2(0, 1),\
			       c02 = F1(0, 2) - F2(0, 2),\
				   c10 = F1(1, 0) - F2(1, 0),\
			       c11 = F1(1, 1) - F2(1, 1),\
			       c12 = F1(1, 2) - F2(1, 2),\
			       c20 = F1(2, 0) - F2(2, 0),\
			       c21 = F1(2, 1) - F2(2, 1),\
			       c22 = F1(2, 2) - F2(2, 2);

			double b00 = F2(0, 0),\
				   b01 = F2(0, 1),\
				   b02 = F2(0, 2),\
				   b10 = F2(1, 0),\
				   b11 = F2(1, 1),\
				   b12 = F2(1, 2),\
				   b20 = F2(2, 0),\
				   b21 = F2(2, 1),\
				   b22 = F2(2, 2);

			Eigen::VectorXd coeff(4);//Ascending with respect to powers
			coeff(0) = b00*b11*b22 + b01*b12*b20 + b10*b21*b02 - b02*b11*b20 - b01*b10*b22 - b00*b12*b21;
			coeff(1) = c00*b11*b22 + c11*b00*b22 + c22*b00*b11 + c01*b12*b20 + c12*b01*b20 + c20*b01*b12
				     + c10*b21*b02 + c21*b10*b02 + c02*b10*b21 - c02*b11*b20 - c11*b02*b20 - c20*b02*b20
					 - c01*b10*b22 - c10*b01*b22 - c22*b01*b10 - c00*b12*b21 - c12*b00*b21 - c21*b00*b12;
			coeff(2) = c00*c11*b22 + c00*c22*b11 + c11*c22*b00 + c01*c12*b20 + c12*c20*b01 + c01*c20*b12
				     + c10*c21*b02 + c10*c02*b21 + c21*c02*b10 - c02*c11*b20 - c02*c20*b11 - c11*c20*b02
					 - c01*c10*b22 - c01*c22*b10 - c10*c22*b01 - c00*c12*b21 - c00*c12*b21 - c12*c21*b00;
			coeff(3) = c00*c11*c22 + c01*c12*c20 + c10*c21*c02 - c02*c11*c20 - c01*c10*c22 - c00*c12*c21;


			std::vector<double> realRoots = UPolynomial::GetRealRoots(coeff);	
            std::vector<Eigen::Matrix3d> fundamentalMatrix;
			for(Eigen::Index i = 0; i < realRoots.size(); i++)
			{
				double alpha = realRoots.at(i);
				Eigen::MatrixXd F = alpha * F1 + (1-alpha) * F2;

				if(F(2, 2) > 1.0e-10)
				{
					Eigen::Matrix3d _f_matrix = F / F(2, 2);
					fundamentalMatrix.push_back(_f_matrix);
				}

			}
            
            
            return fundamentalMatrix;
        }


		void UFundamentalMatrix::NormalizeImagePoints(Eigen::Matrix2d& P, Eigen::Matrix2d& Q, Eigen::MatrixXd& NM)
		{
			//1.Calculate the centroid of all points this is to be estimated
			Eigen::Vector2d centroid(0.0f, 0.0f);
			for(Eigen::Index i = 0; i < P.cols(); i++)
			{
				centroid = centroid + P.col(i);
			}
			centroid = centroid / P.cols();


			//2.Calculate the average distance from all image points to the centroid
			double _average_distance = 0.0f;
			for(Eigen::Index i = 0; i < P.cols(); i++)
			{
				_average_distance += (P.col(i) - centroid).squaredNorm();
			}
			_average_distance = _average_distance / P.cols();
			double nf = std::sqrt(2.0f / _average_distance);//Normalization factor
			double c0 = centroid(0);
			double c1 = centroid(1);
			//Normalization matrix
			NM.resize(3, 3);
			NM <<  nf, 0,  -nf*c0,   
			       0,  nf, -nf*c1,
				   0,  0,   1;//Homogeneous coordinates

			
			//3.Normalize image points
			Q.resize(P.rows(), P.cols());
			for(Eigen::Index i = 0; i < P.cols(); i++)
			{
				Eigen::VectorXd _point = P.col(i); 
				Eigen::VectorXd _normed_point = NM * _point;
				_normed_point = _normed_point / _normed_point(2);

				Q.col(i) = _normed_point.head(2);
			}



		}

		//Normalized eight point method for estimating fundamental matrices
        std::vector<Eigen::Matrix3d> UFundamentalMatrix::SolveBy8PointMethod(Eigen::Matrix2d& P1, Eigen::Matrix2d& P2)
        {
            std::vector<Eigen::Matrix3d> fundamentalMatrix(0);
			//Check the validity of input data
			if(P1.cols() < 8 || P2.cols() < 8 || P1.cols() != P2.cols())
			{
				return fundamentalMatrix;
			}

			Eigen::Matrix2d Q1, Q2;
			Eigen::MatrixXd NM1, NM2;
			this->NormalizeImagePoints(P1, Q1, NM1);
			this->NormalizeImagePoints(P2, Q2, NM2);
			Eigen::MatrixXd A(P1.cols(), 9);//Initialize the coefficients of Af = 0
			
			//Create homegeneous linear equations for eight point method
			//Please refer to Page 279, Multiview Geometry in Computer Vision (Second Edition)
			//By Rechard Hartley, 2004.
			for(Eigen::Index i = 0; i < P1.cols(); i++)
			{
				A.block(i, 0, 1, 3) = P1.col(i).homogeneous() * P2.coeff(i, 0);
				A.block(i, 3, 1, 3) = P1.col(i).homogeneous() * p2.coeff(i, 1);
				A.block(i, 6, 1, 3) = P1.col(i).homogeneous();
			}

			Eigen::JacobSVD<Eigen::MatrixXd> solverA(A, Eigen::ComputeFullV);
			Eigen::MatrixXd Z = solverA.matrixC.col(8);//Be sure that f is row major, but this program is col major.
			Z.resize(3, 3);
			Z = Z.transpose();

			Eigen::JacobSVD<Eigen::MatrixXd> solverZ(Z, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::VectorXd singular = solverZ.singularValues();
			singular(2) = 0.0f;//Must satisfy the constraint that Rank(Z) = 2

			Eigen::Matrix3d F = solverZ.MatrixU() * singular.asDiagonal() * solverZ.MatrixV();

			fundamentalMatrix.push_back(NM2.transpose() * F * NM1.transpose());


   			return fundamentalMatrix; 
        }
    }
}


