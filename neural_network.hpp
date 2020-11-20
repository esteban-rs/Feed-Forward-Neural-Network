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
        vector <vector<double>> targets;

        void get_info();

    friend class Neural_NewtworkFF;
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

        double AcumulateDelta(int i, Layer &nextlayer);

    public:
        Layer (double num_inputs, double num_neurons);
        ~Layer ();

        double get_num_neurons();
        vector <double> outputs;

        void calculate(vector <double> &input);
        void calculateDeltaOutputs(vector<double> &targets);
        void CalculateDeltaHidden(Layer &nextlayer);
        void updateWeigths(vector <double> &inputs, double eta);
};

class Neural_NewtworkFF {
    private:
        // By default 1 hidden and output layer
        int num_layers   = 2;
        int num_inputs   = 0;
        int total_epochs = 0;

        vector <Layer>  Layers;
        vector <double> error;

        int error_compare(double error, DataSet &mydataset);
        void get_error_patter(int i, DataSet &mydataset);

    public:
        Neural_NewtworkFF(int hidden_layers, DataSet &dataset);
        ~Neural_NewtworkFF();

        vector <double> cum_error;

        void train_online(int epochs, double tol, double eta, DataSet &mydataset);
        void test(int epochs, int tol, DataSet &mydataset);
        void show_training_set(DataSet &mydataset);
        void show_info();

};

#endif