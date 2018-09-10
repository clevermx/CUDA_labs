#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <time.h>  


#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
using namespace std;
const int class_col = 4;
__constant__ const int CLASS_COUNT = 3;
const int n = 128;
using namespace std;
const char*  DATASET_NAME = "D:\\Projects\\datasets\\iris.data";
const char* TEST_NAME = "D:\\Projects\\datasets\\iris_test.data";
__constant__ const int ATR_COUNT = 4;
double * data_arr;
unsigned int * res_clases;
int* orig_classes;
pair<int, int> * dist;
double new_inst[ATR_COUNT];
__constant__ const int POINTS_COUNT = n;

__constant__ const int NEIGHBOR = 10;
__constant__ const int BLOCK_SIZE_X = 2; // atributes per block

__constant__ const int BLOCK_SIZE_Y = 64; // points per block
__constant__ const int BLOCK_SIZE_SORT = 4;

__constant__ double  new_point[ATR_COUNT];
int classNum(char* className) {
	if (strcmp(className, "Iris-setosa")) return 1;
	if (strcmp(className, "Iris-versicolor")) return 2;
	if (strcmp(className, "Iris-virginica")) return 3;
	return 0;
}
int read_data(string file_path, int N, int atr_count) {
	ifstream fin(file_path);
	char* buff;
	int k = 0;
	buff = new char[20];
	try {
		if (!fin.is_open())
			cout << "cann't open\n";
		else
		{
			data_arr = new double[N*atr_count];
			orig_classes = new int[N];
			while (!fin.eof() && k<N) {				
				for (int i = 0; i < atr_count; i++)
				{
					fin.getline(buff, 20, ',');
					data_arr[k*atr_count +i] = atof(buff);
					//	cout << data_arr[k][i] << "     "; 
				}
				fin.getline(buff, 20, '\n');
				orig_classes[k] = classNum(buff);			
				k++;
				//cout << buff << endl; 

			}
			fin.close();
		}
	}
	catch (exception ex) {
		fin.close();
		cout << ex.what() << endl;
		delete[] orig_classes;
		delete[] data_arr;
		delete[] buff;
	}
	cout << "final" << endl;

	delete[] buff;
	return k;
}
void checkArray(int * _arr, int arr_size) {
	bool sorted = true;
	for (int i = 1; i < arr_size; i++) {
		if (_arr[i] < _arr[i - 1]) {
			sorted = false;
			break;
		}
	}
	std::wcout << "Array sorting check, sorted: " << std::boolalpha << sorted << std::endl;
}



__constant__ const int GRID_SIZE_X = ATR_COUNT  / BLOCK_SIZE_X;
__constant__ const int GRID_SIZE_SORT = POINTS_COUNT /2/BLOCK_SIZE_SORT;

__constant__ const int GRID_SIZE_Y = POINTS_COUNT/BLOCK_SIZE_Y;
void pause() {
	std::wcout << "Press enter to continue . . . " << std::endl;
	std::cin.ignore();
}

bool inline cudaErrorOccured(cudaError_t _cudaStatus) {
	if (_cudaStatus != cudaSuccess) {
		std::wcout << std::endl << std::endl
			<< "------------------------------"
			<< "CUDA error: " << _cudaStatus << std::endl;
		std::wcout << cudaGetErrorString(_cudaStatus) << std::endl;
		std::wcout
			<< "------------------------------"
			<< std::endl << std::endl;
		return true;
	}
	return false;
}
__device__ bool D_SORTED = false;

__global__ void dist_kernel(double * arr, pair<int, int>* dist) {
	//get own index
	int my_atribute = blockIdx.x * blockDim.x + threadIdx.x;
	int my_point = blockIdx.y * blockDim.y + threadIdx.y;
	if (my_point >= POINTS_COUNT) return;
	if (my_atribute >= ATR_COUNT) return;
	int atr_step = gridDim.x*blockDim.x;
	//calc
	
	double sum = 0;
	double dif;
	
	for (my_atribute; my_atribute < ATR_COUNT; my_atribute+= atr_step ) {
		dif = arr[my_point*ATR_COUNT+ my_atribute] - new_point[my_atribute];
		sum += dif*dif;
	
	}
	
	atomicAdd(&dist[my_point].first, 100*sum);
	
}
__device__ inline void swap(int* arr, int i, int j) {
	int tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;
}
__global__ void even_kernel(pair<int,int> * arr) {
	//get own index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//array for swapping
	__shared__ int shared_arr_dist[BLOCK_SIZE_SORT * 2];
	__shared__ int shared_arr_ind[BLOCK_SIZE_SORT * 2];
	//copying forth	
	int last_deduction = 0;
	if (threadIdx.x == 0) {
		if (blockIdx.x == GRID_SIZE_SORT - 1) last_deduction = 1;
		for (int i = 0; i < blockDim.x * 2 - last_deduction; i++) {
			shared_arr_dist[i] = arr[2 * idx + i + 1].first;
			shared_arr_ind[i] = arr[2 * idx + i + 1].second;
		}
	}
	__syncthreads();
	// Last kernel shouldn't work in this case
	if (idx == BLOCK_SIZE_SORT * GRID_SIZE_SORT - 1) return;
	//swapping
	if (shared_arr_dist[threadIdx.x * 2] > shared_arr_dist[threadIdx.x * 2 + 1]) {
		swap(shared_arr_dist, threadIdx.x * 2, threadIdx.x * 2 + 1);
		swap(shared_arr_ind, threadIdx.x * 2, threadIdx.x * 2 + 1);
		D_SORTED = false;
	}
	__syncthreads();
	//copying back
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2 - last_deduction; i++) {
			arr[2 * idx + i + 1].first = shared_arr_dist[i];
			arr[2 * idx + i + 1].second = shared_arr_ind[i];
		}
	}
}
__global__ void odd_kernel(pair<int, int> * arr) {
	//get own index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//array for swapping
	__shared__ int shared_arr_dist[BLOCK_SIZE_SORT * 2];
	__shared__ int shared_arr_ind[BLOCK_SIZE_SORT * 2];
	//copying forth		
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2; i++) {
			shared_arr_dist[i] = arr[2 * idx + i].first;
			shared_arr_ind[i] = arr[2 * idx + i].second;
		}
	}
	__syncthreads();
	//swapping
	if (shared_arr_dist[threadIdx.x * 2] > shared_arr_dist[threadIdx.x * 2 + 1]) {
		swap(shared_arr_dist, threadIdx.x * 2, threadIdx.x * 2 + 1);
		swap(shared_arr_ind, threadIdx.x * 2, threadIdx.x * 2 + 1);
		D_SORTED = false;
	}
	__syncthreads();
	//copying back
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2; i++) {
			arr[2 * idx + i ].first = shared_arr_dist[i];
			arr[2 * idx + i ].second = shared_arr_ind[i];
		
		}
	}
}

__global__ void vote_kernel(pair<int, int>* dist, int* orig_classes, unsigned int* class_count) {
	int my_point = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned int shared_classes[CLASS_COUNT];
	if (my_point >= NEIGHBOR) return;
	for (int i = my_point; i < NEIGHBOR; i = i + gridDim.x*blockDim.x) {
		atomicInc(&shared_classes[orig_classes[dist[i].second]],1);
	}
	__syncthreads();
	for (int i = 0; i < CLASS_COUNT; i++) {
		atomicAdd(&class_count[i], shared_classes[i]);
	}

	
}
void oddevensort(pair<int, int> * arr, int arr_size) {
	bool sorted = false;
	cudaError_t cudaStatus = cudaSuccess;
	int counter = 0;
	while (!sorted) {
		sorted = true;
		cudaStatus = cudaMemcpyToSymbol(D_SORTED, &sorted, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return;
		odd_kernel << <GRID_SIZE_SORT, BLOCK_SIZE_SORT >> >(arr);
		even_kernel << <GRID_SIZE_SORT, BLOCK_SIZE_SORT >> >(arr);
		cudaStatus = cudaMemcpyFromSymbol(&sorted, D_SORTED, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return;
		counter++;
	}
	std::cout << "Sorting finished, iterations: " << counter << std::endl;
}
void KNN(double *points,int* orig_classes, pair<int, int>* dist,unsigned int * res_classes) {
	cudaError_t cudaStatus = cudaSuccess;
	dim3 Dg = dim3(GRID_SIZE_X, GRID_SIZE_Y);
	dim3 Db = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dist_kernel << <Dg, Db >> > (points, dist);
	if (cudaErrorOccured(cudaStatus)) return;
	oddevensort(dist, POINTS_COUNT);
	if (cudaErrorOccured(cudaStatus)) return;
	 
	vote_kernel<<<GRID_SIZE_X, BLOCK_SIZE_X >>>(dist,orig_classes, res_classes);

	if (cudaErrorOccured(cudaStatus)) return;
}

int main()
{
	int const REP_COUNT = 4;
	clock_t start, stop;
	double sec_copy_in = 0;
	double sec_copy_out = 0;
	double sec_implement = 0;
	cudaError_t cudaStatus = cudaSuccess;
	int arr_size = 0;
	

	//GPU copies of arrays
	double* d_arr = 0; 
	unsigned int * d_res_classes = 0;
	int* d_orig_classes=0;
	pair<int, int> * d_dist=0;
	 //0. Decvice info
	std::wcout << "CUDA realization of odd-even sorting algorithm" << std::endl;
	std::wcout << "CUDA information" << std::endl;
	int deviceCount = 0;
	cudaStatus = cudaGetDeviceCount(&deviceCount);
	if (cudaErrorOccured(cudaStatus)) return 1;
	std::wcout << "Available CUDA device count: " << deviceCount << std::endl << std::endl;
	cudaDeviceProp devProps;
	for (int i = 0; i < deviceCount; i++) {
		cudaStatus = cudaGetDeviceProperties(&devProps, i);
		if (cudaErrorOccured(cudaStatus)) return 1;
		std::wcout
			<< "Device #" << i << ", CUDA version: " << devProps.major << "." << devProps.minor
			<< ", integrated: " << std::boolalpha << devProps.integrated << std::endl
			<< "Name: " << devProps.name << std::endl
			<< "Clockrate: " << (double)devProps.clockRate / 1024 << "MHz" << std::endl
			<< "Total global memory: " << (double)devProps.totalGlobalMem / 1024 / 1024 / 1024 << "GB" << std::endl
			<< "Shared memory per block: " << (double)devProps.sharedMemPerBlock / 1024 << "KB" << std::endl
			<< "Warp size: " << devProps.warpSize << std::endl
			<< "Max threads per block: " << devProps.maxThreadsPerBlock << std::endl
			<< "Max threads dimension: ["
			<< devProps.maxThreadsDim[0] << ", "
			<< devProps.maxThreadsDim[1] << ", "
			<< devProps.maxThreadsDim[2] << "]" << std::endl
			<< "Max grid size: ["
			<< devProps.maxGridSize[0] << ", "
			<< devProps.maxGridSize[1] << ", "
			<< devProps.maxGridSize[0] << "]" << std::endl
			<< std::endl;
	}
	

	for (int rep = 0; rep <= REP_COUNT; rep++) {
		std::wcout << std::endl;
		//1. create host data
		arr_size = read_data(DATASET_NAME,n,class_col);
		dist = new pair<int, int>[n];
		for (int i = 0; i < n; i++) {
			dist[i].first = 0;
			dist[i].second = i;
		}
		
		//point for classification
		new_inst[0] = 6.7;
		new_inst[1] = 3.0;
		new_inst[2] = 5.0;
		new_inst[3] = 1.7;

		res_clases = new unsigned int[CLASS_COUNT];
		for (int i = 0; i < CLASS_COUNT; i++) {
			res_clases[i] = 0;
		}
		std::wcout << "Array generated, size: " << arr_size << std::endl;
		if (rep > 0)	start = clock();
		//2. create device data
		
		 //copy new-point to constant memory
		cudaStatus = cudaMemcpyToSymbol(new_point, new_inst,ATR_COUNT* sizeof(double));
		if (cudaErrorOccured(cudaStatus)) return 1;
		 // copy all points
		cudaStatus = cudaMalloc((void **)&d_arr, arr_size * class_col*sizeof( double));
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMemcpy(d_arr, data_arr, arr_size * class_col * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMalloc((void **)&d_orig_classes, arr_size *sizeof(int));
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMemcpy(d_orig_classes, orig_classes, arr_size *  sizeof(int), cudaMemcpyHostToDevice);
		if (cudaErrorOccured(cudaStatus)) return 1;
		 //copy flag for sort
		cudaStatus = cudaMalloc((void **)&D_SORTED, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return 1;
		 //copy distances
		cudaStatus = cudaMalloc((void **)&d_dist, arr_size * sizeof(pair<int,int>));
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMemcpy(d_dist, dist, arr_size * sizeof(pair<int, int>), cudaMemcpyHostToDevice);
		if (cudaErrorOccured(cudaStatus)) return 1;

		 // copy res_class counts
		cudaStatus = cudaMalloc((void **)&d_res_classes, 3 * sizeof(unsigned int));
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMemcpy(d_res_classes, res_clases, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaErrorOccured(cudaStatus)) return 1;
		if (rep > 0) {
			stop = clock();
			sec_copy_in += (double)(stop - start)*1000.0 / (CLK_TCK*1.0);
		}
		std::wcout << "Memory allocation and copying host->device finished" << std::endl;
		//3. KNN
		if (rep > 0)	start = clock();
		KNN(d_arr,d_orig_classes ,d_dist,d_res_classes);
		cudaStatus = cudaGetLastError();
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaDeviceSynchronize();
		if (cudaErrorOccured(cudaStatus)) return 1;
		if (rep > 0) {
			stop = clock();
			sec_implement += (double)(stop - start)*1000.0 / (CLK_TCK*1.0);
		}

		//4. Get results from device
		if (rep > 0)	start = clock();
		cudaStatus = cudaMemcpy(res_clases, d_res_classes, CLASS_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaFree(d_arr);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaFree(d_dist);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaFree(d_res_classes);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaDeviceReset();;
		if (cudaErrorOccured(cudaStatus)) return 1;
		if (rep > 0) {
			stop = clock();
			sec_copy_out += (double)(stop - start)*1000.0 / (CLK_TCK*1.0);
		}
		std::wcout << "Copying device->host and memory releasing finished" << std::endl;
		//5. check data

		int max_class = -1;
		int max_val = 0;
		for (int i = 0; i < CLASS_COUNT; i++) {
			if (res_clases[i] > max_val) {
				max_class = i;
				max_val = res_clases[i];
			}
		}
		delete[] dist;
		delete[] res_clases;
		cout << "Finished itertion. Result " << max_class << "  " <<endl;
	}
	delete[] data_arr;
	sec_copy_in = sec_copy_in / (REP_COUNT*1.0);
	sec_copy_out = sec_copy_out / (REP_COUNT*1.0);
	sec_implement = sec_implement / (REP_COUNT*1.0);
	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
	std::cout << "=====================================================================" << std::endl;
	std::cout << "*********************************************************************" << std::endl;
	std::cout << "Avarege copying to device: " << sec_copy_in << " msec " << std::endl;
	std::cout << "Avarege implementation : " << sec_implement << " msec " << std::endl;
	std::cout << "Avarege copying from device: " << sec_copy_out << " msec " << std::endl;
	std::cout << "*********************************************************************" << std::endl;
	std::cout << "=====================================================================" << std::endl;
	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
	std::wcout << "Program finished" << std::endl;
	pause();
	return 0;
}