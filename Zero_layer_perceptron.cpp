#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>

struct TrainingData
{
   bool label;
   float coordinates[2];
   TrainingData(const float xy[2],const bool& color):label(color)
   {
        for(int i = 0; i < 2;++i)
        {
            this->coordinates[0] = xy[0];
            this->coordinates[1] = xy[1];
        }    
   }; 
   TrainingData(const bool& color):label(color) {
    for(int i = 0; i < 2;++i)
    (color == 1)?this->coordinates[i] = (pow(-1,i))*(float)rand()/(float)RAND_MAX : this->coordinates[i] = -(float)rand()/(float)RAND_MAX;
   };

};
using Data = std::vector<TrainingData>;
Data Dataset;
void GenerateData(const bool& color)
{
    Dataset.push_back(TrainingData(color));
    return;
}
struct Weight
{
    float coordinates[2];

    Weight()
    {
        this->coordinates[0] = ((float)rand()/(float)RAND_MAX);
        this->coordinates[1] = -((float)rand()/(float)RAND_MAX);
    };

};  
using Weights = std::vector<Weight>;
Weights wey;
void GenerateWeight()
{
    wey.push_back(Weight());
    return;
}
float operator*(Weight w,TrainingData x)
{
    float temp = w.coordinates[0]*x.coordinates[0] + w.coordinates[1]*x.coordinates[1];
    return temp;
}
float operator*(TrainingData x, Weight w)
{
    float temp = w.coordinates[0]*x.coordinates[0] + w.coordinates[1]*x.coordinates[1];
    return temp;
}
bool classification(const TrainingData& dat, const Weight& w)
{
    float temp = dat*w;
    if (temp > 0)return 1;
    else return 0;
}
bool sampleclasif(const int& Datanum)
{
    for (int i = 0; i < Datanum; ++i)
    {
        if(classification(Dataset[i],wey[0])!=Dataset[i].label) return 0;
    }
    return 1;
}

int main()
{   
    int Datanum = 10, epochs = 0;
    GenerateWeight(); //initializing weights
    for (int i = 0; i < Datanum; ++i)
    {
        GenerateData(1); // generating training data for group 1
        GenerateData(0); // generating training data for group 0
    }
    while(sampleclasif(Datanum) != 1) //checking convergance on the training dataset
    {
        for(auto it = Dataset.begin(); it != Dataset.end(); ++it) //training loop
        {
            if(it->label == 1 && wey[0]*(*it) < 0)
            {
                wey[0].coordinates[0] = wey[0].coordinates[0] + it->coordinates[0];
                wey[0].coordinates[1] = wey[0].coordinates[1] + it->coordinates[1];
            }
            else if (it->label == 0 && wey[0]*(*it) >= 0)
            {
                wey[0].coordinates[0] = wey[0].coordinates[0] - it->coordinates[0];
                wey[0].coordinates[1] = wey[0].coordinates[1] - it->coordinates[1];
            }
        }
        ++epochs; //loop counter
    }
    std::cout <<"The final values for w are w[0] = "<<wey[0].coordinates[0]<<" and w[1] = "<<wey[0].coordinates[1]<<std::endl;
    std::cout <<"The model trained for "<<epochs<< " epochs. \n";

    //Creating a new test case
    float array[2] = {-120000.0f,-110000.3f};
    TrainingData testdata(array,false);
    //Testing the model on the case
    std::cout <<"The test case is classified as "<<classification(testdata,wey[0])<<std::endl;

    return 0; 
}