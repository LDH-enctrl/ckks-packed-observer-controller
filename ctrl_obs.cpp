// get tcp_protocol decription
#include "tcp_protocol_client.h" 
// #include "tcp_protocol_client_windows.h" 

// get other tools
#include <iostream>
// #include <cstdint> //if use Windows environment, then active this.
#include <string>
#include <chrono>
using namespace std;

// get model description
#include "model_obs.h"

// init tcp host and port
const string host = "172.31.240.1";
const int port = 9999;

int main()
{
    // set simulation(this section have to set same with plant)
    double sampling_time = 0.02;
    bool run_signal = true;

    // set tcp client
    tcp_client tccp = tcp_client(host, port);
    string signal;

    // for check cycle time
    auto stc = chrono::high_resolution_clock::now();
    auto edc = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(edc - stc);
    double run_time;

    // input/output initialization
    vector<double> y(2, 0.0);
    vector<double> u(1, 0.0);

    obs_plain obs;

    while(run_signal)
    {
        signal = tccp.Recv<string>();

        if(signal == "run")
        {
            // start clock set
            stc = chrono::high_resolution_clock::now();

            // get plant output
            y[0] = tccp.Recv<double>();
            y[1] = tccp.Recv<double>();
            
            // send control input data
            tccp.Send<double>(u[0]);

            // y and u value encryption after packing
            obs.state_update(y);
            u[0] = obs.get_output();

            cout << "y = [" << y[0] << ", " << y[1]
                 << "], u = " << u[0] << endl;

            // end clock set
            edc = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::nanoseconds>(edc - stc);
            run_time = duration.count() / 1000000;
            cout << "loop time: " << run_time << "ms" << endl;

        }
        else if(signal == "end")
        {
            // end of loop signal get
            run_signal = false;
            break;
        }
    }

    return 0;
}
