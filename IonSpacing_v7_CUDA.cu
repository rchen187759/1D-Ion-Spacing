/*
Title: Finding Eqilibrium Positions of Ions in a Linear Chain
Version: 7
Author: Renyi Chen
Description: This program solves equilibrium positions of ions in a linear
	chain by guessing the values then slightly adjust it until convergence

	Tested up to: 1500

CUDA Version: Cuda compilation tools, release 11.0, V11.0.221
*/

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include <chrono>

#define N 1000	//Number of Ions
#define THREAD_PER_BLOCK 32

void guess_gen(double guess[]);
void converge_test(double& alpha, double u_guess[], double u_calc[],
	double u_guess_backup[], double& residual_s, bool& converge);

//Partially Calculation
__global__ void uj_calc_block(double* u, double* u_block) {
	__shared__ double partial_sum[THREAD_PER_BLOCK];

	for (int j = 0; j < N / 2; j++) {
		int index = blockIdx.x * THREAD_PER_BLOCK + threadIdx.x;
		if ((j != index) && (index < N)) {
			if (j > index) {
				partial_sum[threadIdx.x] = (1 / ((u[j] - u[index]) * (u[j] - u[index])));
			}
			else {
				partial_sum[threadIdx.x] = -(1 / ((u[j] - u[index]) * (u[j] - u[index])));
			}
		}
		else {
			partial_sum[threadIdx.x] = 0;
		}

		__syncthreads();
		__syncthreads();

		for (int s = 1; s < blockDim.x; s *= 2) {
			if ((threadIdx.x % (2 * s) == 0) && ((threadIdx.x + s) < blockDim.x)) {
				partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			u_block[j * (N / THREAD_PER_BLOCK + 1) + blockIdx.x] = partial_sum[0];
		}
		__syncthreads();
	}
}

//Obtain New Values from the Partial Calculation
__global__ void uj_calc(double* u_block, double* uj_calc) {
	int calcIndex = blockIdx.x * 32 + threadIdx.x;

	uj_calc[calcIndex] = 0;
	uj_calc[N - calcIndex - 1] = 0;
	if (calcIndex < N / 2) {
		for (int i = 0; i < (N / (THREAD_PER_BLOCK)+1); i++) {
			uj_calc[calcIndex] += u_block[calcIndex * (N / THREAD_PER_BLOCK + 1) + i];
		}

		uj_calc[N - calcIndex - 1] = -uj_calc[calcIndex];
	}
}

int main() {

	//====================================
	auto start = std::chrono::high_resolution_clock::now();
	//====================================

	//====================================
	//host variables
	double* u_guess_h;
	double* u_calc_h;

	//device variables
	double* u_guess_d;
	double* uj_block_result;
	double* u_calculated_d;
	//====================================
	//host memory allocation
	u_guess_h = (double*)malloc(sizeof(double) * N);
	u_calc_h = (double*)malloc(sizeof(double) * N);

	//device memory allocation
	cudaMalloc((void**)&u_guess_d, sizeof(double) * N);
	cudaMalloc((void**)&uj_block_result, sizeof(double) * N * (N / (THREAD_PER_BLOCK)+1) / 2);
	cudaMalloc((void**)&u_calculated_d, sizeof(double) * N);
	//====================================

	//====================================
	//host variables
	double u_temp[N] = { 0 }, u_guess_backup[N] = { 0 };
	bool converge = false;
	double starting_alpha, alpha, residual_s;

	starting_alpha = 0.1;
	alpha = starting_alpha;
	residual_s = 0;
	//====================================

	//====================================
	//generate initial guesses
	guess_gen(u_guess_h);
	//====================================
	//int iteration = 0;
	std::cout << "N = " << N << std::endl;
	//====================================
	while (!(converge)) {
		//iteration++;
		//std::cout << "iteration = " << iteration << std::endl;
		//saving guess backup
		for (int i = 0; i < N; i++) {
			u_guess_backup[i] = u_temp[i];
			u_temp[i] = u_guess_h[i];
		}

		cudaMemcpy(u_guess_d, u_guess_h, sizeof(double) * N, cudaMemcpyHostToDevice);
		uj_calc_block << <(N / THREAD_PER_BLOCK + 1), THREAD_PER_BLOCK >> > (u_guess_d, uj_block_result);
		uj_calc << <N / 64 + 1, 32 >> > (uj_block_result, u_calculated_d);
		cudaMemcpy(u_calc_h, u_calculated_d, sizeof(double) * N, cudaMemcpyDeviceToHost);

		converge_test(alpha, u_guess_h, u_calc_h, u_guess_backup, residual_s, converge);

		//if guess value enters a loop where convergence can't be achieve
		//then the calculation is restarted with smaller starting alpha
		if ((alpha < 1e-12) && (residual_s > 100)) {
			guess_gen(u_guess_h);
			residual_s = 0;
			starting_alpha = starting_alpha * 0.95;
			alpha = starting_alpha;
		}
		else if (alpha < 1e-18) {
			guess_gen(u_guess_h);
			residual_s = 0;
			starting_alpha = starting_alpha * 0.95;
			alpha = starting_alpha;
		}

		/*
		std::cout << "u[0] = " << std::setprecision(16) << u_guess_h[0] << std::endl;
		std::cout << "alpha = " << alpha << std::endl;
		std::cout << "residual_s = " << residual_s << std::endl;
		std::cout << "-----------------------------------------" << std::endl;
		*/
	}
		//====================================
		for (int i = 0; i < N; i++) {
			std::cout << std::setprecision(16) << u_calc_h[i] << ',';
		}
	
		//====================================
		//free host memory
		free(u_guess_h);

		//free device memory
		cudaFree(u_guess_d);
		cudaFree(uj_block_result);
		cudaFree(u_calculated_d);
		//====================================

		//====================================
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		double duration_sec;
		duration_sec = duration;
		duration_sec = duration_sec * 1e-6;
		std::cout << '\n' << duration_sec << std::endl;
		//====================================
	
	return 0;
}

/*=================================================================
Title: converge_test
	Description: check to see if convergence has occure, if not
		modify the guess value accordiang to calculated values
	return: none
=================================================================*/

void converge_test(double& alpha, double u_guess[], double u_calc[],
	double u_guess_backup[], double& residual_s, bool& converge) {

	//=====================================================
	double difference = 1e-8;	//Solution Resolution
	double residual_calc = 0;
	double guess_calc_diff;

	double outOfRange = (u_guess[0] - u_calc[0]) * alpha;	//Variable to check precision limit
	//=====================================================

	//=====================================================
	//calculate the sum of guessing value and calculated value
	for (int i = 0; i < N / 2; i++) {
		guess_calc_diff = std::abs(u_guess[i] - u_calc[i]);
		residual_calc += guess_calc_diff;
	}

	//check for difference between calculated value and guessing value
	for (int i = 0; i < N; i++) {
		double check = (std::abs(std::abs(u_guess[i]) - std::abs(u_calc[i])));
		if (check >= difference) {
			break;
		}
		else if (i == (N - 1)) {
			converge = true;
			return;
		}
	}

	//=====================================================

	//=====================================================
	//if calculated residual is larger than the residual from last iteration
	//alpha is decreased, and guessing value is restored
	if ((residual_s != 0) && (residual_calc > residual_s)) {
		alpha = alpha * 0.9;

		for (int i = 0; i < N; i++) {
			u_guess[i] = u_guess_backup[i];
		}

		return;
	}
	//=====================================================

	//=====================================================
	//reset alpha, when alpha is too small, 
	//which makes the (difference * alpha) too small
	if ((residual_calc == residual_s) && (u_guess[0] == (u_guess[0] - outOfRange))) {
		alpha = 0.1;
	}
	//=====================================================

	//=====================================================
	//if residual is decreased and program did not converge
	//then the guess = alpha * guess + (1-alpha) * calculated value
	for (int i = 0; i < (N / 2); i++) {
		guess_calc_diff = (u_guess[i] - u_calc[i]) * alpha;
		u_guess[i] = u_guess[i] - guess_calc_diff;
	}
	//copied for the other half
	for (int i = 0; i < (N / 2); i++) {
		u_guess[N - 1 - i] = -u_guess[i];
	}
	//=====================================================

	//=====================================================
	//new residual
	residual_s = residual_calc;
	//=====================================================

}

/*=================================================================
Title: guess_gen
	Description: This function generates the initial guess
=================================================================*/
void guess_gen(double guess[]) {
	for (int i = 0; i < N / 2; i++) {
		guess[i] = i - N / 2;
		guess[N - i - 1] = -guess[i];
	}
}


