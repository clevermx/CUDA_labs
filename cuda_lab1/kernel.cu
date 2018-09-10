#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>  

// ---------------------------------
// BEGIN OF USER AREA
// Array size for initialization, used only in inputArray functiont
__constant__ const int G_ARRAY_SIZE = 128;
// Number of threads inside of block
__constant__ const int BLOCK_SIZE = 8;
int inputArray(int ** _arr) {
	int arr_size = G_ARRAY_SIZE;
	*_arr = new int[arr_size];
	for (int i = 0; i < arr_size; i++) {
		(*_arr)[i] = rand() % arr_size;
	}
	/*
	std::wcout << "Array: ";
	for (int i = 0; i < arr_size; i++) {
	std::wcout << (*_arr)[i] << ", ";
	}
	std::wcout << std::endl;
	*/
	return arr_size;
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
// END OF USER AREA
// ---------------------------------
// Number of blocks
__constant__ const int GRID_SIZE = G_ARRAY_SIZE / 2 / BLOCK_SIZE;

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
__device__ inline void swap(int * arr, int i, int j) {
	int tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;
}
__global__ void even_kernel(int * arr) {
	//get own index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
		//array for swapping
	__shared__ int shared_arr[BLOCK_SIZE * 2];
		//copying forth	
	int last_deduction = 0;
	if (threadIdx.x == 0) {
		if ( blockIdx.x == GRID_SIZE - 1) last_deduction = 1;
		for (int i = 0; i < blockDim.x * 2 - last_deduction; i++) {
			shared_arr[i] = arr[2 * idx + i + 1];
		}
	}
	__syncthreads();
	// Last kernel shouldn't work in this case
	if (idx == BLOCK_SIZE * GRID_SIZE - 1) return;
		//swapping
	if (shared_arr[threadIdx.x * 2] > shared_arr[threadIdx.x * 2 + 1]) {
		swap(shared_arr, threadIdx.x * 2, threadIdx.x * 2 + 1);
		D_SORTED = false;
	}
	__syncthreads();
	//copying back
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2 - last_deduction; i++) {
			arr[2 * idx + i + 1] = shared_arr[i];
		}
	}
}
__global__ void odd_kernel(int * arr) {
	//get own index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//array for swapping
	__shared__ int shared_arr[BLOCK_SIZE * 2];
	//copying forth		
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2; i++) {
			shared_arr[i] = arr[2 * idx + i];
		}
	}
	__syncthreads();
	//swapping
	if (shared_arr[threadIdx.x * 2] > shared_arr[threadIdx.x * 2 + 1]) {
		swap(shared_arr, threadIdx.x * 2, threadIdx.x * 2 + 1);
		D_SORTED = false;
	}
	__syncthreads();
	//copying back
	if (threadIdx.x == 0) {
		for (int i = 0; i < blockDim.x * 2; i++) {
			arr[2 * idx + i] = shared_arr[i];
		}
	}
}
__global__ void useless_kernel(int * arr) {
	//get own index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	arr[idx] = arr[idx] + arr[idx];;	
}
int uselessWork() {
	cudaError_t cudaStatus = cudaSuccess;
	int * arr;
	int arr_size = inputArray(&arr);
	int * d_arr = 0; //GPU copy of array
	cudaStatus = cudaMalloc((void **)&d_arr, arr_size * sizeof(int));
	if (cudaErrorOccured(cudaStatus)) return 1;
	cudaStatus = cudaMemcpy(d_arr, arr, arr_size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaErrorOccured(cudaStatus)) return 1;
	useless_kernel <<<GRID_SIZE, BLOCK_SIZE >> >(d_arr);
	if (cudaErrorOccured(cudaStatus)) return 1;
	cudaStatus = cudaGetLastError();
	if (cudaErrorOccured(cudaStatus)) return 1;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaErrorOccured(cudaStatus)) return 1;
	cudaStatus = cudaMemcpy(arr, d_arr, arr_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaErrorOccured(cudaStatus)) return 1;
	cudaStatus = cudaFree(d_arr);
	if (cudaErrorOccured(cudaStatus)) return 1;
	cudaStatus = cudaDeviceReset();;
	if (cudaErrorOccured(cudaStatus)) return 1;	
}
void oddevensort(int * arr, int arr_size) {
	bool sorted = false;
	cudaError_t cudaStatus = cudaSuccess;
	int counter = 0;
	while (!sorted) {
		sorted = true;
		cudaStatus = cudaMemcpyToSymbol(D_SORTED, &sorted, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return;
		odd_kernel << <GRID_SIZE, BLOCK_SIZE >> >(arr);
		even_kernel << <GRID_SIZE, BLOCK_SIZE >> >(arr);
		cudaStatus = cudaMemcpyFromSymbol(&sorted, D_SORTED, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return;
		counter++;
	}
	std::cout << "Sorting finished, iterations: " << counter << std::endl;
}
int main()
{
	int const REP_COUNT = 10;
	clock_t start, stop;
	double sec_copy_in = 0;
	double sec_copy_out = 0;
	double sec_implement = 0;
	cudaError_t cudaStatus = cudaSuccess;
	int arr_size = 0;
	int * arr = 0;
	int * d_arr = 0; //GPU copy of array
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
	//0. useless work
	uselessWork();
	for (int rep = 0; rep <= REP_COUNT; rep++) {
		std::wcout << std::endl;
		//1. create host data
		arr_size = inputArray(&arr);
		std::wcout << "Array generated, size: " << arr_size << ", last element: " << arr[arr_size - 1] << std::endl;
		if (rep > 0)	start = clock();
		//2. create device data
		cudaStatus = cudaMalloc((void **)&D_SORTED, sizeof(bool));
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMalloc((void **)&d_arr, arr_size * sizeof(int));
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaMemcpy(d_arr, arr, arr_size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaErrorOccured(cudaStatus)) return 1;
		if (rep > 0) {
			stop = clock();
			sec_copy_in += (double)(stop - start)*1000.0 / (CLK_TCK*1.0);
		}
		std::wcout << "Memory allocation and copying host->device finished" << std::endl;
		//3. Sorting
		if (rep > 0)	start = clock();
		oddevensort(d_arr, arr_size);
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
		cudaStatus = cudaMemcpy(arr, d_arr, arr_size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaFree(d_arr);
		if (cudaErrorOccured(cudaStatus)) return 1;
		cudaStatus = cudaDeviceReset();;
		if (cudaErrorOccured(cudaStatus)) return 1;
		if (rep > 0) {
			stop = clock();
			sec_copy_out += (double)(stop - start)*1000.0 / (CLK_TCK*1.0);
		}
		std::wcout << "Copying device->host and memory releasing finished" << std::endl;
		//5. check data
		checkArray(arr, arr_size);
		delete[] arr;
		std::wcout << "Array output finished" << std::endl;
	}
	sec_copy_in = sec_copy_in	 / (REP_COUNT*1.0);
	sec_copy_out = sec_copy_out / (REP_COUNT*1.0);
	sec_implement = sec_implement / (REP_COUNT*1.0);
	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
	std::cout << "=====================================================================" << std::endl;
	std::cout << "*********************************************************************" << std::endl;
	std::cout << "Avarege copying to device: " << sec_copy_in << " msec "<< std::endl;
	std::cout << "Avarege implementation : " << sec_implement << " msec " << std::endl;
	std::cout << "Avarege copying from device: " << sec_copy_out << " msec " << std::endl;
	std::cout << "*********************************************************************" << std::endl;
	std::cout << "=====================================================================" << std::endl;
	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
	std::wcout << "Program finished" << std::endl;
	pause();
	return 0;
}