#include<stdio.h>
#include<string.h>
#include<random>
#include<string>
#include<algorithm>
#include<set>

std::mt19937 generator(22566789);
std::uniform_int_distribution<int> dist(0,std::numeric_limits<int>::max());

void dump_func(int cas,int N,int D,int x0,int x1,int A,int B,int C,int M){
	printf("dumping case %d\n",cas);
	std::string file_name = "sample" + std::to_string(cas) + ".in";
	FILE* fp = fopen(file_name.c_str(),"w");
	fprintf(fp,"%d %d %d %d %d %d %d %d\n",N,D,x0,x1,A,B,C,M);
	fclose(fp);
}

int main(){
	dump_func(1,10,3,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(2,100,10,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(3,1000,10,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(4,1000,20,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(5,1000,40,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(6,10000,10,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(7,10000,100,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(8,100000,1200,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(9,100000,1400,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	dump_func(10,100000,1600,dist(generator),dist(generator),dist(generator),dist(generator),dist(generator),1000000007);
	return 0;
}

