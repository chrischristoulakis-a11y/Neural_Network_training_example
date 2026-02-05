#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "Training_data.h"

const int inputnum = 3, hiddennum = 3, epochs = 10000;
const float learning_rate = 0.1/*, l2_rate = 0.05*/;
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
void reset(float (&grad)[inputnum+1][hiddennum+1][2])
{
    for(int i = 0; i <= inputnum; ++i)
    {
        for(int j = 0; j <= hiddennum; ++j)
        {
            for(int k = 0; k < 2; ++k)
            {
                grad[i][j][k] = 0;
            }
        }
    }
    return;
}
void printweights(std::ofstream& prog, const float (&grad)[inputnum+1][hiddennum+1][2], const int& epoch)
{
    prog << "Epoch: "<<epoch<<"\n-----------------------------------\n";
    for (int k = 0; k < 2; ++k)
    {
        for (int i = 0; i <= inputnum; ++i)
        {
            for (int j = 0; j <= hiddennum; ++j)
            {
                prog << std::fixed<<std::setprecision(4)<< grad[i][j][k] <<" ";
            }
            prog << "\n";
        }
        prog << "\n";
    }
    prog << "-----------------------------------------\n";
}
int main()
{   srand(time(NULL));
    std::ofstream prog;
    prog.open("weight_prog.txt");
    float loss = 0;
    for (int i = 0; i < inputnum; ++i)
    {
        inputnodes.push_back(InputNode());       
    }
    for (int i = 0; i < hiddennum; ++i)
    {
        hiddennodes.push_back(HiddenNode());       
    }
    OutputNode out;
    float weights[inputnum+1][hiddennum+1][2];
    for (int i = 0; i < inputnum + 1; ++i)
    {
        for (int j = 0; j < hiddennum + 1; ++j)
        {
            weights[i][j][0] = (2*(float)rand()/(float)RAND_MAX) - 1;
             
        }
        weights[0][i][1] = (2*(float)rand()/(float)RAND_MAX) - 1;
        weights[i][0][0] = 0;
    }
    weights[0][0][1] = (2*(float)rand()/(float)RAND_MAX) - 1;
    weights[0][0][0] = 0;
    float gradients[inputnum+1][hiddennum+1][2];

    float trainingdata[][inputnum] = {INPUT_DATA_LIST};
    float result[] = OUTPUT_DATA_LIST;
    int datanum = sizeof(result)/sizeof(result[0]);
            //Training starts here
    for(int epoch = 0; epoch < epochs; ++epoch){        
    for (int traind = 0; traind < datanum; ++traind)
    {
        for (int i = 0; i < inputnum; ++i) // propagates from input nodes to hidden nodes
        {
            inputnodes[i].output = trainingdata[traind][i];
            for(int j = 0; j < hiddennum; ++j)
            {
                hiddennodes[j].input[i+1] = trainingdata[traind][i]*weights[i+1][j+1][0];
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
        out.error = out.input - result[traind]; //calulates error term for the output node
        if (epoch % (epochs/10) == 0 || epoch == epochs - 1) loss += pow(out.error,2)/datanum;
        for (int i = 0; i < hiddennum; ++i) //calculates error term for the hidden nodes
        {
            hiddennodes[i].error = hiddennodes[i].output*(1 - hiddennodes[i].output)*out.error*weights[0][i+1][1];
        }
        gradients[0][0][1] += out.error/datanum; //output bias diff
        for (int i = 0; i < hiddennum; ++i)
        {
            gradients[0][i+1][0] += hiddennodes[i].error/datanum; //hidden bias diff
            gradients[0][i+1][1] += (out.error*hiddennodes[i].output)/datanum; //calculates partial derivatives for the hidden-output weights
            for (int j = 0; j < inputnum; ++j) //calculates partial derivatives for the input-hidden weights
            {
                gradients[j+1][i+1][0] += (inputnodes[j].output*hiddennodes[i].error)/datanum;
            }
        }
    }
    for (int i = 0; i < hiddennum; ++i)
    {
        for (int j = 0; j < inputnum; ++j)
        {
            weights[j+1][i+1][0] -=learning_rate*gradients[j+1][i+1][0];//updates input-hidden weights
        }
        weights[0][i+1][0] -= learning_rate*gradients[0][i+1][0];//updates hidden biasses
        weights[0][i+1][1] -= learning_rate*gradients[0][i+1][1];//updates hidden-output weights
    }
    weights[0][0][1] -= learning_rate*gradients[0][0][1]; //updates output bias
    if (epoch % (epochs/10) == 0 || epoch == epochs - 1)
    {
        printweights(prog,weights,epoch);
        std::cout << "Epoch: " << epoch << " Loss: "<<loss/2<<"\n";
        loss = 0;
    }
    reset(gradients);}
    prog.close();
    std::ofstream file;
    file.open("Weight_matrix.txt");
    for (int k = 0; k < 2; ++k)
    {
        for (int i = 0; i <= inputnum; ++i)
        {
            for (int j = 0; j <= hiddennum; ++j)
            {
                file <<std::fixed<<std::setprecision(4)<< weights[i][j][k] <<" ";
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
    return 0;
}
