#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include<iomanip>

const int inputnum = 3, hiddennum = 3;
float sigmoid(const float& x)
{
    float temp = 1 /(1 + exp(-x));
    return temp;
}
struct InputNode 
{
    float output;
    InputNode():output(0)
    {
    };
};
std::vector<InputNode> inputnodes = {};
struct HiddenNode
{
    float input[inputnum+1],output,activation,error;
    
    HiddenNode():output(0),activation(0),error(0)
    {
        this->input[0] = 1;
        for (int i = 0; i < inputnum; ++i)
        {
            this->input[i+1] = 0;
        }
    };
    float suminputs()
    {
        float temp = 0;
        for (int i = 0; i < inputnum; ++i)
        {
            temp += this->input[i+1];
        }
        this->activation = temp;
        return temp;
    }
    float fpropagate()
    {
        float temp = sigmoid(this->activation);
        this->output = temp;
        return temp;
    } 
};
std::vector<HiddenNode> hiddennodes = {};
struct OutputNode
{   
    float input,error;
    OutputNode():input(0),error(0){};
};

int main()
{
    for (int i = 0; i < inputnum; ++i)
    {
        inputnodes.push_back(InputNode());       
    }
    for (int i = 0; i < hiddennum; ++i)
    {
        hiddennodes.push_back(HiddenNode());       
    }
    OutputNode out;
    float weights[hiddennum+1][inputnum+1][2];
    std::ifstream reader;
    reader.open("Weight_matrix.txt");
    if (!reader.is_open()) {std::cerr <<"Error opening Weight_matrix.txt.\n";return 1;}
    for(int k = 0; k < 2; ++k)
    {   
        for(int i = 0; i <= inputnum; ++i)
        {
            for(int j = 0; j <= hiddennum; ++j)
            {
                reader >> weights[i][j][k];
            }
        }
    }
    float testdata[][inputnum] = {{0,0,0},{0,0,1},{1,0,0},{0,1,0},{1,1,1}};
    float testnum = sizeof(testdata)/sizeof(testdata[0]);
    std::vector<float> result;
    result.reserve(testnum);
    //Network begins here
    for(int test = 0; test < testnum; ++test)
    {
         for (int i = 0; i < inputnum; ++i) // propagates from input nodes to hidden nodes
        {
            inputnodes[i].output = testdata[test][i];
            for(int j = 0; j < hiddennum; ++j)
            {
                hiddennodes[j].input[i+1] = testdata[test][i]*weights[i+1][j+1][0];
            }
        }
        out.input = 0;
        for (int i = 0; i < hiddennum; ++i) // propagates from hidden nodes to output 
        {
            hiddennodes[i].suminputs();
            hiddennodes[i].activation += weights[0][i][0];
            hiddennodes[i].fpropagate();
            out.input += hiddennodes[i].output*weights[0][i+1][1];
        }
        out.input += weights[0][0][1]; //adds the bias term to the output
        result[test] = out.input;
        std::cout <<"The output of the number "<<test+1<<" test is: "<< result[test]<<"\n";
    }
    return 0;
}