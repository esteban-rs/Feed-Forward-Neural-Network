#include <bits/stdc++.h>
#include "neural_network.hpp"
#include <time.h>

/* ************** N e u r o n  C l a s s ************** */

Neuron::Neuron(double numberOfInputs) {
    // Conections of the Neuron
    number_of_inputs = numberOfInputs;

    // Initialice Weigths
    weigths.assign(numberOfInputs, 0);
    initial_weigths(weigths, numberOfInputs);

    // Select Activation function 
        
}

Neuron::~Neuron(){

}

double Neuron::sigmoid(double x) {
    return 1/( 1 + exp(-x));
}

double Neuron::sigmoidDerivate(double x) {
    return x*(1-x);
}

double Neuron::calculate(vector <double> &input){
    double sum = 0;
    for (int i = 0; i < input.size(); i++) {
        // Ponder Input
        sum += input[i] * weigths[i];
    }

    // Set sum of information
    information_received = sum;
    // Set Output for next layer
    output = sigmoid(information_received);

    return output;
}

void Neuron::calculate_delta_output(double target){
    delta = sigmoidDerivate(output)*(output - target);
}

void Neuron::calculate_delta_hidden(double acummulate_delta){
    delta = sigmoidDerivate(output)*acummulate_delta;
}

void Neuron::get_info() {
    cout << " Number of Inputs: " << number_of_inputs <<"."<< endl;
    cout << " Neuron Information: " << information_received <<"."<< endl;
    cout << "Weigths: " << endl;

    for (int i = 0; i < number_of_inputs; i++){
        cout << "[" << i << "] :" << weigths[i] << endl;
    }
}

double Neuron::get_delta_value() {
    return delta;
}

double Neuron::get_weigth_value(int i){
    return weigths[i];
}

void initial_weigths(vector<double> &weights, int n){
    // weigths between -1 and 1
    //longitud = 2;

    srand48(time(NULL)); // cambio semilla

    for (int i = 0; i < n; i++){
        weights[i] = -1 + drand48()*2;
    }
}

/* ************** L a y e r  C l a s s ************** */

Layer::Layer (double num_inputs, double num_neurons){
    // Set Info
    number_of_inputs  = num_inputs;
    number_of_neurons = num_neurons;

    // Add Neurons to layer and Initialize
    for (int i = 0; i < number_of_neurons; i++){
        Neurons.push_back(Neuron(num_inputs));
        outputs.push_back(0);
    }
}

Layer::~Layer(){
    
}

double Layer::get_num_neurons(){
    return number_of_neurons;
}

void Layer::calculate(vector <double> &input) {
    for (int i = 0; i < number_of_neurons; i++){
        outputs[i] = Neurons[i].calculate(input);
    }
}

void Layer::calculateDeltaOutputs(vector<double> &targets) {
    for (int i = 0; i < number_of_neurons; i++){
        Neurons[i].calculate_delta_output(targets[i]);
    }
}

void Layer::CalculateDeltaHidden(Layer &nextlayer) {
    for (int i = 0; i < number_of_neurons; i++){
        Neurons[i].calculate_delta_hidden(AcumulateDelta(i, nextlayer));
    }
}

double Layer::AcumulateDelta(int i, Layer &nextlayer) {
    double sum = 0;
    for (int j = 0; j < nextlayer.get_num_neurons(); j++){
        sum += nextlayer.Neurons[j].get_delta_value() * nextlayer.Neurons[j].get_weigth_value(i);
    }
    return sum;
}

void Layer::updateWeigths(vector <double> &inputs, double eta) {
    for (int i = 0; i < number_of_neurons; i++){
        for (int j = 0; j < Neurons[i].weigths.size(); j++){
            Neurons[i].weigths[j] += -eta * Neurons[i].delta * inputs[j];
        }
    }
}

/* ************** D a t a S e t  C l a s s ************** */

DataSet::DataSet(string filename) {
    ifstream file(filename);

    if (file.is_open() == 0){
        cerr << "No existe el archivo." <<endl;
        exit(-1);
    }

    string word;
    while (word[0] != 'Y'){
        num_inputs++;
        file >> word;
    }

    // Substract y
    num_inputs -= 1;

    vector <double> tmp;
    tmp.assign(num_inputs,0);

    double y = 0;

    // get data
    int i = 0;
    while (file >> tmp[0]){
        // Read X
        for (i = 1; i < num_inputs; i++){
            file >> tmp[i];
        }
        
        // Read Y
        file >> y;
        data.push_back(tmp);
        targets.push_back({y});
    }

    file.close();
    normalize();
}

DataSet::~DataSet() {

}

void DataSet::get_info() {
    cout << "Inputs:  " << num_inputs << endl;
    cout << "Data for training: " << data.size() << endl;
    cout << "Flags for Normalization " << endl;
}

void DataSet::normalize() {
    // Identify data that requeries normalization
    int flag = 0;
    for (int i = 0; i < num_inputs; i++){
        for (int j = 0; j < data.size(); j++){
            if (fabs(data[j][i]) > 1){
                flag = 1;
                break;
            }
        }
        data_flag_normalize.push_back(flag);
        flag = 0;
    }

    // Normalize 
    double mean = 0;
    double std  = 0;
    int old_num_inputs = num_inputs;

    for (int i = 0; i < old_num_inputs; i++){
        if (data_flag_normalize[i] == 1){
            for (int j = 0; j < data.size(); j++){
                mean += data[j][i];
            }
            mean = mean/data.size();

            // std 
            for (int j = 0; j < data.size(); j++){
                std += (data[j][i] - mean)*(data[j][i] - mean);
            }
            std = sqrt(std/data.size());

            // Normalize
            for (int j = 0; j < data.size(); j++){
                data[j][i] = (data[j][i] - mean)/std;
            }
        }
        // clean mean and std
        mean = 0;
        std  = 0;
    }
    // add Bias
    for (int i = 0; i < data.size(); i++){
        data[i].push_back(1);
    }
    // Increase size for bias
    num_inputs++;
}

/* ************** N e u r a l  N e t w o r k ************** */

Neural_NewtworkFF::Neural_NewtworkFF(int hidden_layers, DataSet &dataset) {
    // set size by console
    int size = 0;
    cout << "Number of Neurons for Layer[0]: ";
    cin >> size;

    // First Hidden Layer
    Layers.push_back(Layer(dataset.data[0].size(), size));

    // Hidden Layers
    for (int i = 1; i < hidden_layers; i++){
        cout << "Number of Neurons for Layer[" << i << "]: ";
        cin >> size;

        Layers.push_back(Layer(Layers[i-1].get_num_neurons(), size));
        num_layers++;
    }
    // Output Layer
    Layers.push_back(Layer(Layers[num_layers-2].get_num_neurons(), dataset.targets[0].size()));

    // Set first layer size
    num_inputs = dataset.data[0].size();

    // error for each epoch in dataset
    error.assign(dataset.targets.size(), 0);
}

Neural_NewtworkFF::~Neural_NewtworkFF() {

}

void Neural_NewtworkFF::train_online(int epochs, double tol, double eta, DataSet &mydataset) {
    int epoch          = 0;
    double local_error = 0;

    int i = 0;
    int j = 0;
    while (epoch <= epochs){
        epoch++;
        local_error = 0;

        for (i = 0; i < mydataset.data.size(); i++) {
            // Calculate Forward
            Layers[0].calculate(mydataset.data[i]);
            for (j = 1; j < num_layers; j++) {
                Layers[j].calculate(Layers[j-1].outputs);
            }

            get_error_patter(i,mydataset);

            // Calculate Backward
            Layers[num_layers-1].calculateDeltaOutputs(mydataset.targets[i]);
            for (j = num_layers-2; j >= 0; j--) {
                Layers[j].CalculateDeltaHidden(Layers[j+1]);
            }
        
            // Modify Weigths
            Layers[0].updateWeigths(mydataset.data[i],eta);
            for ( j = 1; j < num_layers; j++){
                Layers[j].updateWeigths(Layers[j-1].outputs,eta);
            }
        }

        if (error_compare(tol, mydataset) == 1){
                break;
        }
    }
    total_epochs = epoch-1;
}

void Neural_NewtworkFF::get_error_patter(int i, DataSet &mydataset) {
    // Last layer has one neuron 
    error[i] = (Layers[num_layers-1].outputs[0]- mydataset.targets[i][0]);
    error[i] *= error[i];
}


int Neural_NewtworkFF::error_compare(double global_error, DataSet &mydataset) {
    int size = error.size();
    double training_error = 0;

    int flag = 1;
    for (int i = 0; i < size; i++){
        // check all error
        if (error[i] > global_error){
            flag = 0;
        }
        // Acummulated error
        training_error += error[i];
    }
    // Add error to cum_error in class NNFF
    training_error = training_error/mydataset.targets.size();
    cum_error.push_back(training_error);

    return flag;
}

void Neural_NewtworkFF::show_training_set(DataSet &mydataset){
    cout << "----------------------" << endl;
    cout << "* NN val vs Target *" <<endl;
    for (int i = 0; i < mydataset.data.size(); i++){
        Layers[0].calculate(mydataset.data[i]);
        for (int j = 1; j < num_layers; j++){
            Layers[j].calculate(Layers[j-1].outputs);
        }

        for (int j = 0; j < Layers[num_layers-1].outputs.size(); j++){
            cout << setprecision(5) << fixed << Layers[num_layers - 1].outputs[j];
            cout << "      " << setprecision(1) << mydataset.targets[i][0] << endl;
        }
    }
}

void Neural_NewtworkFF::show_info() {
    cout << "----------------------" << endl;

    cout <<"Neurons in Layer[0]: "<< num_inputs << endl;

    for (int i = 0; i < num_layers; i++){
        cout <<"Neurons in Layer["<< i+1 << "]: "<< Layers[i].get_num_neurons() << endl;
    }
    cout << "----------------------" << endl;

    cout << "Epochs for training: " << total_epochs << endl;
    cout << "* Layer[0] has a bias neuron." << endl;
}

void Neural_NewtworkFF::test(int epochs, int tol, DataSet &mydataset) {

}
