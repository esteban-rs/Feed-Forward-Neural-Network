#include <bits/stdc++.h>
#include <time.h>
#include "neural_network.hpp"

// Agregar tiempo 
int main(int argc, char const *argv[]){
    /* *** D a t a S e t  S e t u p *** */
    string filename;
    cout << "Input DataSet Filename:   ";
    cin >> filename;
    cout << endl;

    DataSet mydataset(filename);

    cout << "---- Neural Network Setup ----"<< endl;

    int size = 0;
    cout << "Number of Hidden Layers:  ";
    cin >> size;

    Neural_NewtworkFF mynetwork(size, mydataset);

    cout << endl;
    cout << "------ Parameters Setup ------"<< endl;

    int epochs = 0;
    cout << "Maximum Number of Epochs: ";
    cin >> epochs;

    double tol = 0;
    cout << "Tolerance:                ";
    cin >> tol;

    double eta = 0;
    cout << "Learning rate:            ";
    cin >> eta;
    cout << endl;

    clock_t start = clock();
    
    mynetwork.train_online(epochs, tol, eta, mydataset);
    
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC; 
    
    mynetwork.show_info();
    
    cout << endl << "Execution Time : " << cpu_time_used << "segundos." << endl; 

    /*for (int i = 0; i < mynetwork.cum_error.size(); i++){
        cout << mynetwork.cum_error[i] << endl;
    }*/
    //mynetwork.show_training_set(mydataset);


    return 0;
}