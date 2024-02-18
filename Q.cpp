#include <iostream>
#include <thread>
#include <mutex>
#include <random>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <semaphore>
#include <condition_variable>
#include <fcntl.h>
#include <semaphore.h>
#include <stack>



using namespace std;



class Neuron {
public:
    double value;
    double *weights;
    int num_weights;

    Neuron(int num_weights) : num_weights(num_weights) {
        weights = new double[num_weights];
    }

    // ~Neuron() {
    //     delete[] weights;
    // }
};

class Layer {
public:
    Neuron **neurons;
    int num_neurons;

    Layer(int num_neurons, int num_weights_per_neuron) : num_neurons(num_neurons) {
        neurons = new Neuron *[num_neurons];
        for (int i = 0; i < num_neurons; ++i) {
            neurons[i] = new Neuron(num_weights_per_neuron);
        }
    }

    // ~Layer() {
    //     for (int i = 0; i < num_neurons; ++i) {
    //         delete neurons[i];
    //     }
    //     delete[] neurons;
    // }
};

stack<double> backup;
double temp(){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_real_distribution<double> distribution(0.001, 0.999);
    return (distribution(generator));
}
double val1(double a){
return (a*a+a+1)/2;
}
double val2(double b){
    return (b*b-b)/2;
}

int count=0;
struct ForwardPropArgs {
    Layer *prev_layer;
    Layer *cur_layer;
    int neuron_idx;
    int pipe_fd[2];
    mutex *print_mutex;
    condition_variable *print_cv;
    int *current_print_idx;
};

void *forward_propagation(void *args) {
    ForwardPropArgs *fpa = (ForwardPropArgs *)args;
    Layer *prev_layer = fpa->prev_layer;
    Layer *cur_layer = fpa->cur_layer;
    int neuron_idx = fpa->neuron_idx;
    int *pipe_fd = fpa->pipe_fd;
    mutex *print_mutex = fpa->print_mutex;
    condition_variable *print_cv = fpa->print_cv;
    int *current_print_idx = fpa->current_print_idx;

   

    double sum = 0.0;
    for (int i = 0; i < prev_layer->num_neurons; ++i) {
        sum += prev_layer->neurons[i]->value * prev_layer->neurons[i]->weights[neuron_idx];
        
    }
    if(sum==0)sum=temp();

    double sigmoid = 1.0 / (1.0 + exp(-sum));
    backup.push(sum);
    //count++;

    // Wait for its turn to print
    {
        unique_lock<mutex> lock(*print_mutex);
        print_cv->wait(lock, [&]() { return *current_print_idx == neuron_idx; });

        //cout << "Neuron " << neuron_idx + 1 << " generated sigmoid " << sigmoid << " now writing in pipe for next layer." << endl;
        ++(*current_print_idx);
    }
    print_cv->notify_all();

    write(pipe_fd[1], &sigmoid, sizeof(sigmoid));


    return nullptr;
}



void propagate_layers(Layer *prev_layer, Layer *cur_layer) {
    int num_neurons = cur_layer->num_neurons;
    int pipes[num_neurons][2];
    pthread_t threads[num_neurons];
    ForwardPropArgs args[num_neurons];
    mutex print_mutex;
    condition_variable print_cv;
    int current_print_idx = 0;

    for (int i = 0; i < num_neurons; ++i) {
        if (pipe(pipes[i]) == -1) {
            perror("pipe");
            exit(EXIT_FAILURE);
        }
        cout << "Neuron " << i + 1 << " working." << endl;
        count++;
        args[i] = {prev_layer, cur_layer, i, {pipes[i][0], pipes[i][1]}, &print_mutex, &print_cv, &current_print_idx};
        pthread_create(&threads[i], nullptr, forward_propagation, (void *)&args[i]);
        pthread_join(threads[i], nullptr);
        double activation;
        read(pipes[i][0], &activation, sizeof(activation));
        cur_layer->neurons[i]->value = activation;
        close(pipes[i][0]);

        close(pipes[i][1]);
    }

}


int main() {
    //Create the neural network
    int num_inputs = 2;
    int num_hidden_layers = 5;
    int neurons_per_hidden_layer = 8;
    Layer input_layer(num_inputs, neurons_per_hidden_layer);

    Layer** hidden_layers = new Layer*[num_hidden_layers];

    for (int i = 0; i < num_hidden_layers; ++i) {
        hidden_layers[i] = new Layer(neurons_per_hidden_layer, neurons_per_hidden_layer);
    }


    Layer output_layer(neurons_per_hidden_layer, 1);
    double input_values[] = {0.1, 0.2};

    // Input layer weights
    double input_layer_weights[][8] = {
        {0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.1, -0.2},
        {-0.4, 0.5, 0.6, -0.4, 0.5, 0.6, -0.4, 0.5},
    };

    double hidden_layer_weights[5][8][8] = {
        {
            {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
            {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
            {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
            {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
            {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
            {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
            {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
            {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
        },

        {
            {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
            {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
            {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
            {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
            {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
            {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
            {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
            {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},

        },
        {
            {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
            {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
            {0.6, -0.5, -0.7, 0.2, 0.4, 0.8, -0.1, -0.3},
            {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
            {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
            {0.6, -0.5, -0.7, 0.2, 0.4, 0.8, -0.1, -0.3},
            {0.3, -0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1},
            {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},

        },
        {
            {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
            {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},
            {0.5, -0.4, -0.6, 0.3, 0.2, 0.8, -0.2, -0.1},
            {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
            {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},
            {0.5, -0.4, -0.6, 0.3, 0.2, 0.8, -0.2, -0.1},
            {0.4, -0.5, 0.6, -0.7, -0.8, 0.9, -0.1, 0.2},
            {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2},

        },
        {
            {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
            {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},
            {0.4, -0.3, -0.5, 0.1, 0.6, 0.7, -0.3, -0.2},
            {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
            {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},
            {0.4, -0.3, -0.5, 0.1, 0.6, 0.7, -0.3, -0.2},
            {0.5, -0.6, 0.7, -0.8, -0.9, 0.1, -0.2, 0.3},
            {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1},

        }
    
    };
    double output_layer_weights[] = {-0.1, 0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8};

    for (int i = 0; i < num_inputs; ++i) 
    {
    input_layer.neurons[i]->value = input_values[i];
    }
    //I layers
    for (int i = 0; i < num_inputs; ++i) 
    {
        for (int j = 0; j < neurons_per_hidden_layer; ++j) {
            input_layer.neurons[i]->weights[j] = input_layer_weights[i][j];
        }
    
    }
    //H Layers
    for (int layer = 0; layer < num_hidden_layers; ++layer) 
    {
        for (int i = 0; i < neurons_per_hidden_layer; ++i) {
            for (int j = 0; j < neurons_per_hidden_layer; ++j) {
                
                hidden_layers[layer]->neurons[i]->weights[j]= hidden_layer_weights[layer][i][j];
            }
        }
    }
     sem_t *semaphores[num_hidden_layers - 1];

    // Initialize the semaphores
    // for (int i = 0; i < num_hidden_layers - 1; i++) {
    //     string semaphore_name = "propagate_semaphore_" + to_string(i);
    //     semaphores[i] = sem_open(semaphore_name.c_str(), O_CREAT, 0644, (i == 0) ? 1 : 0);
    //     if (semaphores[i] == SEM_FAILED) {
    //         perror("sem_open");
    //         exit(EXIT_FAILURE);
    //     }
    // }


    pid_t pid[num_hidden_layers - 1];
    cout<<"Input layer propagating"<<endl;
    propagate_layers(&input_layer, hidden_layers[0]);

   
    for (int i = 0; i < num_hidden_layers - 1; i++) {
        //sem_wait(semaphores[i]);
       // pid[i]=fork();
       // if(pid[i]==0){
            cout<<"layer "<<i+1<<" in propagating"<<endl;
            propagate_layers(hidden_layers[i], hidden_layers[i + 1]);
        //    if (i < num_hidden_layers ) {
        //         sem_post(semaphores[i + 1]); // Release the next semaphore
        //     }

            //exit(EXIT_SUCCESS);
        //}
        
       
    }
    
    // for(int i=0; i<num_hidden_layers-1; i++){
    //     wait(&pid[i]);
    // }
    //  for (int i = 0; i < num_hidden_layers - 1; i++) {
    //     string semaphore_name = "propagate_semaphore_" + to_string(i);
    //     sem_close(semaphores[i]);
    //     sem_unlink(semaphore_name.c_str());
    // }


    cout<<"Output layer propagating"<<endl;
    propagate_layers(hidden_layers[num_hidden_layers - 1], &output_layer);
    double ans=output_layer.neurons[0]->value;
    cout << "Output: " << ans << endl;
    int x=6;
    while(!backup.empty()){
        cout<<"Layer "<<x<<endl;
        for(int i=0; i<8; i++){
            cout<<"Neuron "<<i+1<<" Value "<<backup.top()<<endl; 
            backup.pop();
        }
        x--;
    }
    
    cout<<count<<endl;


    return 0;
}

