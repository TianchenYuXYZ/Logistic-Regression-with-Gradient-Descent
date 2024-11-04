#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include<vector>
#include<random>

int N,D,x0,x1,A,B,C,M;

template<class T>
T* generate_input(int x0,int x1, int A,int B,int C,int M,size_t size){
  T* ret = new T[size];
  ret[0] = x0 % M;
  ret[1] = x1 % M;
  for(int i = 2;i < size;++i)
    ret[i] = (long long)((long long)A * ret[i - 1] + (long long)B * ret[i - 2] + C) % M;
  return ret;
}

double dotproduct(double* X,int D,double* P){
  double ret = 0;
  for(int i = 0;i < D;++i)
    ret += X[i] * P[i];
  return ret;
}

double sigmoid(double X){
  return 1.0 / (1 + exp(-X));
}

double* init_parameters(int D){
  double* ret = new double[D];
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);
  for(int i = 0;i < D;++i){
    ret[i] = distribution(generator);
  }
  return ret;
}

double forward(double* X,int D,double* P){
  double logit = dotproduct(X,D,P);
  double pred = sigmoid(logit);
  return pred;
}

double get_loss(double* X,int N,int D,int* Y,double* P){
  double total_loss = 0;
  for(int i = 0;i < N;++i){
    double pred = forward(X + i * D,D,P);
    double loss = Y[i] * log(pred) + (1 - Y[i]) * log(pred);
    total_loss += loss;
  }
  total_loss = -1.0 / N * total_loss;
  return total_loss;
}

double get_acc(double* X,int N,int D,int* Y,double* P){
  int correct = 0;;
  for(int i = 0;i < N;++i){
    double pred = forward(X + i * D,D,P);
    if(round(pred) == Y[i])
      ++correct;
  }
  return correct * 1.0 / N;
}

double learning_rate_schedule(int epoch){
  if(epoch < 10000)
    return 0.1;
  else if(epoch < 20000)
    return 0.01;
  return 0.001;
}

int* generate_label(double* X,int N,int D){
  int* ret = new int[N];
  for(int i = 0;i < N;++i){
    double tmp = 0;
    for(int j = 0;j < D;++j){
      if(j % 2)
        tmp += X[i * D + j] * X[D + (j + (int)tmp * 9999) % D];
      else
        tmp -= X[i * D + j] * X[D + (j + (int)tmp * 10001) % D];
    }
    ret[i] = round(sigmoid(tmp));
  }
  return ret;
}

int main(int argc,char** argv){
  FILE* fin = fopen(argv[1],"r");
  FILE* fout = fopen(argv[2],"r");
  fscanf(fin,"%d%d%d%d%d%d%d%d",&N,&D,&x0,&x1,&A,&B,&C,&M);
  fclose(fin);
  double* X = generate_input<double>(x0,x1,A,B,C,M,(size_t)N * D);
  for(int i = 0;i < (size_t)N * D;++i)
    X[i] = X[i] * 1.0 / M;
  int* Y = generate_label(X,N,D);
  double* P = new double[D];
  for(int i = 0;i < D;++i){
    fscanf(fout,"%lf",&P[i]);
  }
  fclose(fout);
  double tolerance = 0.6;

  double acc = get_acc(X,N,D,Y,P);
  printf("got acc %f\n",acc);
  if(acc >= tolerance){
    printf("correct\n");
    return 0;
  }else{
    printf("incorrect\n");
    return 1;
  }
}