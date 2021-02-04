/*
Title: Finding Eqilibrium Positions of Ions in a Linear Chain
Version: 7
Author: Renyi Chen
Description: This program solves equilibrium positions of ions in a linear
	chain by guessing the values then slightly adjust it until convergence

	Tested up to: 1500

*/


#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>

const int N = 275;	//Number of Ions

double uj_calc(int j, double u_old[]);
bool guessing(double u_guess[], double& alpha, double u_guess_backup[], double& residual_s);
void guess_gen(double guess[]);

int main() {

	//******************************************
	//Wall Clock Time
	auto start = std::chrono::high_resolution_clock::now();
	//******************************************

	double u_guess[N] = { 0 }, u_temp[N] = { 0 }, u_guess_backup[N] = { 0 };
	bool converge = false;
	double starting_alpha, alpha, residual_s;

	starting_alpha = 0.1;
	alpha = starting_alpha;
	residual_s = 0;

	//generate initial guess
	guess_gen(u_guess);

	std::cout << "N = " << N << std::endl;

	while (not(converge)) {

		//save guess backup
		for (int i = 0; i < N; i++) {
			u_guess_backup[i] = u_temp[i];
			u_temp[i] = u_guess[i];
		}

		converge = guessing(u_guess, alpha, u_guess_backup, residual_s);

		//if guess value enters a loop where convergence can't be achieve
		//then the calculation is restarted with smaller starting alpha
		if ((alpha < 1e-12) && (residual_s > 100)) {
			guess_gen(u_guess);
			residual_s = 0;
			starting_alpha = starting_alpha * 0.95;
			alpha = starting_alpha;
		}
		else if (alpha < 1e-18) {
			guess_gen(u_guess);
			residual_s = 0;
			starting_alpha = starting_alpha * 0.95;
			alpha = starting_alpha;
		}

		/*
		//For Debugging
		std::cout << "u[0] = " << std::setprecision(16) << u_guess[0] << std::endl;
		std::cout << "alpha = " << alpha << std::endl;
		std::cout << "residual_s = " << residual_s << std::endl;
		std::cout << "-----------------------------------------" << std::endl;
		*/
	}

	//Display solutions
	for (int i = 0; i < N; i++) {
		std::cout << std::fixed << std::setprecision(15) << u_guess[i] << ", ";
	}

	//**********************************
	//Wall Clock Tim
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	double duration_sec;
	duration_sec = duration;
	duration_sec = duration_sec * 1e-6;
	std::cout << '\n' << duration_sec << std::endl;
	//**********************************
	return 0;
}

/*
Title: uj_calc
Description: This function Rearrange the equilibrium equation to uj=...
	and calculates uj using guess values
	J = 0,1,2,...,N-1
*/
double uj_calc(int j, double u_old[]) {
	double first_sum = 0, sec_sum = 0;

	for (int i = 0; i < j; i++) {
		first_sum += (1 / pow(u_old[j] - u_old[i], 2));
	}

	for (int i = j + 1; i < N; i++) {
		sec_sum += (1 / pow(u_old[j] - u_old[i], 2));
	}

	return (first_sum - sec_sum);
}

/*
Title: guessing
Description: This function calculates for convergence
*/

bool guessing(double u_guess[], double& alpha, double u_guess_backup[], double& residual_s) {

	double difference = 1e-8;	//resolution of the solution
	double u_oldL[N] = { 0 };

	double residual_calc = 0;

	//calculate new values from guessing values
	for (int j = 0; j < (N / 2); j++) {
		u_oldL[j] = uj_calc(j, u_guess);
	}

	//since answers are symmetric, first half is negated and copied
	for (int i = 0; i < (N / 2); i++) {
		u_oldL[N - 1 - i] = -u_oldL[i];
	}

	//calculate the residual of the guessing value and calculated value
	for (int i = 0; i < (N / 2); i++) {
		residual_calc += pow((u_guess[i] - u_oldL[i]), 2);
	}

	//check for difference between calculated value and guessing value
	for (int i = 0; i < N; i++) {
		double check = (std::abs(std::abs(u_guess[i]) - std::abs(u_oldL[i])));
		if (check >= difference) {
			break;
		}
		else if (i == (N - 1)) {
			return true;
		}
	}

	//if calculated residual is larger than the residual from last iteration
	//alpha is decreased, and guessing value is restored
	if ((residual_s != 0) && (residual_calc > residual_s)) {
		alpha = alpha * 0.9;

		for (int i = 0; i < N; i++) {
			u_guess[i] = u_guess_backup[i];
		}

		return false;
	}

	//if reached precision limit
	double outOfRange = (u_guess[0] - u_oldL[0]) * alpha;
	if ((residual_calc == residual_s) && (u_guess[0] == (u_guess[0] - outOfRange))) {
		alpha = 0.1;
	}


	//if residual is decreased and program did not converge
	//then the guess = alpha * guess + (1-alpha) * calculated value
	for (int i = 0; i < (N / 2); i++) {
		double percent = (u_guess[i] - u_oldL[i]) * alpha;
		u_guess[i] = u_guess[i] - percent;
	}

	//copied for the other half
	for (int i = 0; i < (N / 2); i++) {
		u_guess[N - 1 - i] = -u_guess[i];
	}

	//new residual
	residual_s = residual_calc;
	return false;
}

/*
Title: guess_gen
	Description: This function generates the initial guess
*/

void guess_gen(double guess[]) {


	for (int i = 0; i < N / 2; i++) {
		guess[i] = i - N / 2;
		guess[N - i - 1] = -guess[i];
	}
}