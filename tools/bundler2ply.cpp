#include<iostream>
#include<fstream>
//#include"optimization/parallel_bundle_adjustment.hpp"

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cout<<"Please enter bundler out file for translation."<<std::endl;
    }
    std::ifstream in(argv[1]);

    int nCamera, nStructure, nProjection;
    //Read bundler out file
    if(!(in>>nCamera>>nStructure>>nProjection))
    {
	    return 0; 
    }

    std::ofstream of("point_cloud.ply");
	of<<"ply"<<std::endl;
	of<<"format ascii 1.0"<<std::endl;
	of<<"element vertex "<<nStructure<<std::endl;
	of<<"property double x"<<std::endl;
	of<<"property double y"<<std::endl;
	of<<"property double z"<<std::endl;
	of<<"end_header"<<std::endl;

    for(unsigned int i = 0; i < nProjection; i++)
	{
		int camidx, optidx;
		double x, y;

		if(!(in>>camidx>>optidx>>x>>y))
			return 0;
    }

    for(unsigned int i = 0; i < nCamera; i++)
    {
        double p[9];
		for(int j = 0; j < 9; j++) in>>p[j];
        //of<<p[3]<<"  "<<p[4]<<"  "<<p[5]<<std::endl;
    }

    for(unsigned int i = 0; i < nStructure; i++)
    {
        double p0, p1, p2;
		in>>p0>>p1>>p2;
        of<<p0<<"  "<<p1<<"  "<<p2<<std::endl;
    }

    in.close();
    of.close();
}
