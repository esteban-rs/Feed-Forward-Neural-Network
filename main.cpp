#include <bits/stdc++.h>
#include <time.h>
#include "neural_network.hpp"

int main(int argc, char const *argv[]){
    
    /*vector<double> weigths;
    weigths.assign(10,0);

    generate_points(weigths,10);

    for (int i = 0; i < 10; i++){
        cout <<weigths[i] << endl;
    }*/
    DataSet mydataset("data.dat");
    mydataset.get_info();
    
    return 0;
}