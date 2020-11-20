#include <bits/stdc++.h>
#include <time.h>
#include "neural_network.hpp"

int main(int argc, char const *argv[]){
    /* *** D a t a S e t  S e t u p *** */
    string filename;
    cout << "Input DataSet Filename: ";
    cin >> filename;

    DataSet mydataset(filename);


    /* *** N e t w o r k  S e t u p *** */
    int size = 0;
    cout << "Number of Hidden Layers: ";
    cin >> size;

    Neural_NewtworkFF mynetwork(size, mydataset);

    int epochs = 0;
    cout << "Maximum Number of Epochs: ";
    cin >> epochs;

    double tol = 0;
    cout << "Tolerance: ";
    cin >> tol;

    double eta = 0;
    cout << "Leargin rate: ";
    cin >> eta;

    mynetwork.train_online(epochs, tol, eta, mydataset);

    mynetwork.show_info();

    for (int i = 0; i < mynetwork.cum_error.size(); i++){
        cout << mynetwork.cum_error[i] << endl;
    }
    //mynetwork.show_training_set(mydataset);


    return 0;
}