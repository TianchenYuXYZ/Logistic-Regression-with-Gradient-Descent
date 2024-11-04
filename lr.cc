#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <iostream>

int N, D, x0, x1, A, B, C, M;

template<class T>
T* generate_input(int x0, int x1, int A, int B, int C, int M, size_t size) {
    T* ret = new T[size];
    ret[0] = x0 % M;
    ret[1] = x1 % M;
    for (int i = 2; i < size; ++i)
        ret[i] = (long long)((long long)A * ret[i - 1] + (long long)B * ret[i - 2] + C) % M;
    return ret;
}

int dynamic_batch_size(int N){
  if (N >= 1000) {
    return 32; 
  } else if (N >= 500) {
    return 64;
  } else if (N >= 100) {
    return 128;
  } else {
    return 256;
  }

}

double dotproduct(double* X, int D, double* P) {
    double ret = 0;

    // #pragma omp parallel for reduction(+:ret)
    for (int i = 0; i < D; ++i) ret += X[i] * P[i];
    return ret;
}

double sigmoid(double X) {
    return 1.0 / (1 + exp(-X));
}

double* init_parameters(int D) {
    double* ret = new double[D];
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < D; ++i) ret[i] = distribution(generator);
    return ret;
}

double forward(double* X, int D, double* P) {
    double logit = dotproduct(X, D, P);
    return sigmoid(logit);
}

double get_loss(double* X, int N, int D, int* Y, double* P) {
    double total_loss = 0;

    #pragma omp parallel for reduction(+:total_loss)
    for (int i = 0; i < N; ++i) {
        double pred = forward(X + i * D, D, P);
        double epsilon = 1e-15;
        pred = std::min(std::max(pred, epsilon), 1 - epsilon);  // Prevent log(0)
        total_loss += Y[i] * log(pred) + (1 - Y[i]) * log(1 - pred);
    }
    total_loss = -1.0 / N * total_loss;
    return total_loss;
}

double get_acc(double* X, int N, int D, int* Y, double* P) {
    int correct = 0;

    // # pragma omp parallel for reduction(+:correct)
    for (int i = 0; i < N; ++i) {
        double pred = forward(X + i * D, D, P);
        if (round(pred) == Y[i]) ++correct;
    }
    return correct * 1.0 / N;
}

double learning_rate_schedule(int epoch) {
    if (epoch < 200) return 0.01;
    else if (epoch < 500) return 0.001;
    return 0.0001;
}

int* generate_label(double* X, int N, int D) {
    int* ret = new int[N];

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double tmp = 0;
        for (int j = 0; j < D; ++j) {
            if (j % 2) tmp += X[i * D + j] * X[D + (j + (int)tmp * 9999) % D];
            else tmp -= X[i * D + j] * X[D + (j + (int)tmp * 10001) % D];
        }
        ret[i] = round(sigmoid(tmp));
    }
    return ret;
}

// Grid search
std::vector<int> batch_sizes = {16, 32, 64, 128, 256};
std::vector<double> learning_rates = {0.0001, 0.001, 0.01, 0.1};

int main(int argc, char** argv) {

    // CONTROL NUMBER OF THREADS
    omp_set_num_threads(8);

    FILE* fin = fopen(argv[1], "r");
    FILE* fout = fopen(argv[2], "w");
    fscanf(fin, "%d%d%d%d%d%d%d%d", &N, &D, &x0, &x1, &A, &B, &C, &M);
    fclose(fin);
    double* X = generate_input<double>(x0, x1, A, B, C, M, (size_t)N * D);
    for (int i = 0; i < (size_t)N * D; ++i) X[i] = X[i] * 1.0 / M;
    int* Y = generate_label(X, N, D);
    double tolerance = 0.6;

    double best_accuracy = 0.0;
    int best_batch_size = 0;
    double best_learning_rate = 0.0;
    double* best_parameters = new double[D];

    // Perform grid search
    for (int batch_size : batch_sizes) {
        for (double lr : learning_rates) {
            double* P = init_parameters(D);

            double* grad = new double[D];
            double current_accuracy = 0.0;
            // double* pred_value = new double[N];

            for (int epoch = 0; epoch < 500; ++epoch) {
                // Declare the variable to store the runtime for this epoch
                // double epoch_runtime = 0.0;
                
                // Start timing the epoch
                // auto epoch_start = std::chrono::high_resolution_clock::now();

                for (int batch_start = 0; batch_start < N; batch_start += batch_size) {
                    int batch_end = std::min(batch_start + batch_size, N);
                    int current_batch_size = batch_end - batch_start;

                    // Initialize gradient for this batch
                    for (int i = 0; i < D; ++i) {
                        grad[i] = 0;
                    }

                    // Parallel gradient computation for the mini-batch
                    #pragma omp parallel num_threads(8)
                    {
                        // Thread-local gradients
                        double* thread_grad = new double[D];
                        for (int i = 0; i < D; ++i) thread_grad[i] = 0;

                        #pragma omp for schedule(static) 
                        for (int i = batch_start; i < batch_end; ++i) {
                            double pred_value = forward(X + i * D, D, P);
                            for (int j = 0; j < D; ++j) {
                                thread_grad[j] += 1.0 / current_batch_size * (pred_value - Y[i]) * X[i * D + j];
                            }
                        }

                        // Reduce the gradients across threads
                        #pragma omp critical
                        {
                            for (int i = 0; i < D; ++i) {
                                grad[i] += thread_grad[i];
                            }
                        }
                        delete[] thread_grad;
                    }

                    // Update parameters after processing the mini-batch
                    //Regularization
                    double regularization_strength = 0.01;

                    for (int i = 0; i < D; ++i) {
                        P[i] -= learning_rate_schedule(epoch) * (grad[i] + regularization_strength * P[i]);
                    }
                }

                // Measure runtime and accuracy for this epoch
                // auto epoch_end = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
                // epoch_runtime = epoch_duration.count();

                // Loss and accuracy check
                double loss = get_loss(X, N, D, Y, P);

                
                double acc = get_acc(X, N, D, Y, P);

                // Print the accuracy and runtime for this epoch
                // printf("Epoch %d: Accuracy = %f, Runtime = %f seconds\n", epoch, acc, epoch_runtime);
                // fflush(stdout);

                // if (acc > tolerance) {
                //     printf("Done with acc %f at epoch %d\n", acc, epoch);
                //     fflush(stdout);
                //     break;
                // }
            }

            // Check accuracy after this epoch
            double acc = get_acc(X, N, D, Y, P);
            if (acc > best_accuracy) {
                best_accuracy = acc;
                best_batch_size = batch_size;
                best_learning_rate = lr;

                // Copy current parameters to best_parameters
                std::copy(P, P + D, best_parameters);


            }


                
    

            delete[] P;
            delete[] grad;
        }
    }

    // Output best parameters found from grid search
    printf("Best batch size: %d, Best learning rate: %.5f, Best accuracy: %.5f\n", best_batch_size, best_learning_rate, best_accuracy);
    for (int i = 0; i < D; ++i){
        fprintf(fout, "%.6f\n", best_parameters[i]);
    }
                    

    fclose(fout);
    delete[] X;
    delete[] Y;
    delete[] best_parameters;
    return 0;
}
