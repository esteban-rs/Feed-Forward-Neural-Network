#ifndef FFNN_HPP
#define FFNN_HPP

using namespace std;

class DataSet {
    private:
        int num_inputs = 0;
        
        vector <int> data_flag_normalize;
        int targets_data_normalize;

        void normalize();
        
    public:
        DataSet(string filename);
        ~DataSet();
        
        vector <vector<double>> data;
        vector <double>         targets;

        void get_info();


};

class Neuron {
    private:
        double number_of_inputs     = 0;
        double information_received = 0;
        double output               = 0; 

        // weigths of Neuron
        vector <double> weigths;

        // Activate Function
        double sigmoid(double x);

        // Derivate
        double delta = 0;
        double sigmoidDerivate(double x);

        // Calculations
        double calculate(vector <double> &input);
        void calculate_delta_output(double target);
        void calculate_delta_hidden(double acummulate_delta);

    public:
        // Constructor/Destructor
        Neuron(double numberOfInputs);
        ~Neuron();

        void get_info();

        double get_delta_value();
        double get_weigth_value(int i);

        /* Check latter
        void Activate_Function(string func);
        */
    
    // layer needs weigths and outputs
    friend class Layer;
};

void initial_weigths(vector<double> &weights, int n);

class Layer {
    private:
        double number_of_inputs  = 0;
        double number_of_neurons = 0;

        vector <Neuron> Neurons;
        vector <double> outputs;

        double AcumulateDelta(int i, Layer &nextlayer);

    public:
        Layer (double num_inputs, double num_neurons);
        ~Layer ();

        double get_num_neurons();
        
        void calculate(vector <double> &input);
        void calculateDeltaOutputs(vector<double> &targets);
        void CalculateDeltaHidden(Layer &nextlayer);
        void updateWeigths(vector <double> &inputs);
};

class Neural_NewtworkFF {
    private:
        
    public:
};

#endif