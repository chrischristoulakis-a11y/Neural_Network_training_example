#include <cmath>
#include <fstream>
#include <vector>
#include <array>
#include <iomanip>
int main()
{
    int data_per_dim = 10;
    const size_t dim = 3;
    float temp;
    std::array<float,dim> coo;
    std::vector <std::array<float,3>> coord;
    std::vector <float> datares;
    for(int i = 0; i < data_per_dim; ++i)
    {
        coo[0] = i;
        for(int j = 0; j < data_per_dim; ++j)
        {
            coo[1] = j;
            for(int k = 0; k < data_per_dim; ++k)
            {
                coo[2] = k;
                coord.push_back(coo);
                temp = (pow(i,2) + pow(j,2) + pow(k,2))/1000;
                datares.push_back(temp);
            }
        }
    }
    std::ofstream file;
    file.open("Training_data.txt");
    for(int i = 0; i < pow(data_per_dim,(int)dim); ++i)
    {
        file << "{";
        for(int j = 0; j < (int)dim; ++j)
        {
            file << coord[i][j];
            if(j < ((int)dim-1)) file << ",";
        }
        file << "}";
        if (i < pow(data_per_dim,(int)dim) - 1) file << ",";
    }
    file << std::endl;
    file << "{";
    for(int i = 0; i < pow(data_per_dim,(int)dim); ++i)
    {
        file <<std::setprecision(4)<< datares[i];
        if (i < pow(data_per_dim,(int)dim) - 1) file << ",";
    }
    file << "}";
    file.close();
    return 0;
}
