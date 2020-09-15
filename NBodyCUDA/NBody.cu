#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include "NBody.h"
#include "NBodyVisualiser.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USER_NAME "aca15npm"		//replace with your username
#define SIZE 256

void print_help();
void step(void);
int numOfBodies;
u_char dimension;
MODE op_mode;
int iterations;
char* filename;
nbody* allBodies;        // nbody array for all simulation bodies
FILE* input_file;
int fileUsed = 0;        // is a CSV file being used? 0 (false) by default
float* heatDensities;   // heat activity map

// device variables/pointers (used as __shared__ variables to store in the GPU cache)
__shared__ nbody* d_allBodies;
__shared__ float* d_heatDensities;

int main(int argc, char* argv[]) {
	// use the current time to seed the random value generator
	srand(time(0));

	// create timers but only 1 will be used depending on op_mode specified
	time_t start = clock();            // create timer for CPU mode
	cudaEvent_t start_cuda, stop_cuda;  // and for GPU mode

	// GPU timer requires event records
	cudaEventCreate(&start_cuda);
	cudaEventCreate(&stop_cuda);
	cudaEventRecord(start_cuda);  // start cuda timer

	if (argc < 4 || argc % 2 != 0 || argc > 8) {
		printf("Insufficient or incorrect arguments given to run the program. Please re-run with arguments as follows: \n");
		printf("\n");
		print_help();
		exit(1);
	}

	else if (argc >= 4) {
		// CHECK THAT PARAMS ARE GIVEN SENSIBLE VALUES. EXIT IF NOT.
		if (atoi(argv[1]) < 1) {  // make sure at least 1 body is specified for simulation
			printf("Invalid number of bodies specified. Please make sure you specify at least 1 body.\n");
			exit(1);
		}
		else if (atoi(argv[2]) < 1) {  // make sure dimension is at least 1
			printf("Invalid dimension specified. Please make sure you specify a dimension of at least 1.\n");
			exit(1);
		}
		// make sure OPMODE is either CPU or OpenMP
		if (strcmp(argv[3], "CPU") == 0) {
			op_mode = CPU;
		}
		else if (strcmp(argv[3], "OPENMP") == 0) {
			op_mode = OPENMP;
		}
		else if (strcmp(argv[3], "CUDA") == 0) {
			op_mode = CUDA;
		}
		else {
			printf("Invalid operation mode. Please make sure you specify either CPU or OPENMP as an operation mode.\n");
			exit(1);
		}
		// Once checks pass, assign param values to variable
		numOfBodies = atoi(argv[1]);
		dimension = atoi(argv[2]);
	}


	allBodies = (nbody*)malloc(sizeof(nbody) * numOfBodies);  // allocate host memory for allBodies

	if (argc >= 6) {
		for (int i = 4; i < argc; i += 2) {
			// Loop through the optional arguments, check that flags are correct and that neither is entered twice
			// if an iteration flag exists, get the number of iterations
			if (strcmp(argv[i], "-i") == 0 && strcmp(argv[i], argv[i - 2]) != 0) {
				if (atoi(argv[i + 1]) < 1) {
					printf("Invalid number of iterations specified. Please specify an interation number of at least 1.\n");
					exit(1);
				}
				else {
					iterations = atoi(argv[i + 1]);
				}
			}
			// if a CSV file flag exists, open it and extract the nbodies for the simulation
			else if (strcmp(argv[i], "-f") == 0 && strcmp(argv[i], argv[i - 2]) != 0) {
				filename = argv[i + 1];
				input_file = fopen(filename, "r");

				if (input_file == NULL) {  // check that the file actually exists...
					printf("Specified file not found. Please check that you have entered the directory and/or filename correctly.\n");
					exit(1);
				}
				else {
					fileUsed = 1;			   // set global variable to indicate file is being used
					char fileLine[250];        // allocate reasonable size to read in each file line
					u_int noOfInputBodies = 0; // this will keep track of how many bodies exist in the input file
					while (!feof(input_file)) {
						fgets(fileLine, 250, input_file);
						if (fileLine[0] != '#') {
							char* commaPtr;					 // pointer for each comma
							char value[15];					 // string to store each extracted value
							float value_f;					 // value converted to float
							nbody body = { 0, 0, 0, 0, 0 };  // create new body for each line read in

							// init variables for finding number values
							char from = 0;
							char to = 0;
							commaPtr = strchr(fileLine, ',');

							for (int i = 0; i < 5; i++) {  // file should contain 4 commas so iterate until the 4th comma is located

								// if it's already found all 4 commas then look for the line terminator instead
								if (commaPtr == NULL) {
									to = strchr(fileLine, '\0') - fileLine;
								}
								else {
									to = strchr(commaPtr, ',') - fileLine;
								}

								// calculate to/from indexes of next value between commas
								strncpy(value, (fileLine + from), (to - from));
								value[to - from] = '\0';

								// if 2 commas are next to each other or there is no value, generate a default value
								if (to - from < 2) {
									// check which field the value belongs to and assign default value accordingly
									if (i == 0 || i == 1) {
										value_f = (float)rand() / (float)(RAND_MAX);  // starting positions
									}
									else if (i == 2 || i == 3) {  // velocity components
										value_f = 0;
									}
									else if (i == 4) {  // mass
										value_f = 1 / numOfBodies;
									}
								}
								else {
									value_f = atof(value);  // else take the string value from file and convert to float
								}

								// finally, move on to the next comma if not already on the last one
								if (commaPtr != NULL) {
									commaPtr += 1;
									from = commaPtr - fileLine;
									commaPtr = strchr(commaPtr, ',');
								}

								// assign values to the body fields depending on which field it's on
								if (i == 0) {	   // x start position
									body.x = value_f;
								}
								else if (i == 1) { // y start position
									body.y = value_f;
								}
								else if (i == 2) { // x velocity
									body.vx = value_f;
								}
								else if (i == 3) { // y velocity
									body.vy = value_f;
								}
								else if (i == 4) { // mass
									body.m = value_f;
								}
							}
							allBodies[noOfInputBodies] = body;  // add body to the array of nbodies
							// and increment body count to keep track of how many are extracted from the CSV file
							noOfInputBodies += 1;
						}
					}

					fclose(input_file);  // close file once no longer needed

					if (numOfBodies != noOfInputBodies) {
						printf("The number of bodies you have specified does not match the number of bodies in the input file. Please check your arguments and try again.\n");
						exit(1);
					}
				}
			}

			else if (strcmp(argv[i], argv[i - 2]) == 0) {
				printf("It appears you have entered an optional argument flag twice. Please re-run with arguments as follows: \n");
				printf("\n");
				print_help();
				exit(1);
			}

			else { // else advise the user that something is wrong
				printf("There appears to be an issue with the arguments you have given, such as not providing the correct flags. Please re-run with arguments as follows: \n");
				printf("\n");
				print_help();
				exit(1);
			}
		}
	}

	if (fileUsed == 0) {  // if CSV file wasn't used, generate bodies with random values
		for (int i = 0; i < numOfBodies; i++) {
			// Generate a set of random data for bodies
			float x = (float)rand() / (float)(RAND_MAX);
			float y = (float)rand() / (float)(RAND_MAX);
			float vx = 0.0f;
			float vy = 0.0f;
			float m = (float)1 / (float)numOfBodies;

			// create body object with values generated
			nbody body = { x, y, vx, vy, m };
			allBodies[i] = body;  // assign body to next position in body array
		}
	}

	// Start simulation here
	// If in CUDA mode, then the device variable for allBodies on the GPU needs to be setup. Otherwise this part can be just skipped
	if (op_mode == CUDA) {
		// copy over from host variable
		cudaMalloc((void**)&d_allBodies, (sizeof(nbody) * numOfBodies));
		cudaMemcpy(d_allBodies, allBodies, (sizeof(nbody) * numOfBodies), cudaMemcpyHostToDevice);
	}

	// start visualiser if no iteration argument was provided
	if (iterations == 0) {
		heatDensities = (float*)malloc(sizeof(float) * dimension * dimension);

		if (op_mode == CUDA) {
			// create heatDensities variable for device and copy over from host variable
			cudaMalloc((void**)&d_heatDensities, (sizeof(float) * dimension * dimension));
			cudaMemcpy(d_heatDensities, heatDensities, (sizeof(float) * dimension * dimension), cudaMemcpyHostToDevice);

			// start visualiser with CUDA mode
			initViewer(numOfBodies, dimension, op_mode, step_cuda);
			setNBodyPositions(d_allBodies);
			setHistogramData(d_heatDensities);
			startVisualisationLoop();
			cudaFree(d_heatDensities);  // free memory on GPU used for device heatmap
		}
		else {
			// start visualiser in non-CUDA mode
			initViewer(numOfBodies, dimension, op_mode, step);
			setNBodyPositions(allBodies);
			setHistogramData(heatDensities);
			startVisualisationLoop();
		}

		free(heatDensities);  // free memory used for host heatmap
	}
	else {
		// if a number of iterations are specified then visualiser is not needed. Call only step function
		if (op_mode == CUDA) {
			// if in CUDA mode, configure the grid of thread blocks and run the kernel
			for (int c = 0; c < iterations; c++) {
				step_cuda();
			}
			cudaFree(d_allBodies);  // free memory on GPU used for device bodies
		}
		else {
			// otherwise run the step() function for the number of iterations given
			for (int c = 0; c < iterations; c++) {
				step();
			}
		}

	}

	free(allBodies);  // free memory used for allBodies

	time_t end = clock(); // end timer
	// get time in milliseconds and print out execution time
	u_int time_spent = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Execution time %d seconds and %d milliseconds \n", (time_spent / 1000), (time_spent % 1000));

	return 0;
}


void step_cuda(void) {
	// Perform the main simulation of the NBody system per iteration
	dim3 threadsPerBlock(SIZE);
	dim3 blocksPerGrid(numOfBodies / SIZE + 1);
	UpdateValues_Cuda << <blocksPerGrid, threadsPerBlock >> > (d_allBodies, numOfBodies);

	if (iterations == 0) {
		cudaMemset(d_heatDensities, 0.0f, (sizeof(float) * dimension * dimension));
		CountHeatDensitiesCuda << <blocksPerGrid, threadsPerBlock >> > (d_allBodies, d_heatDensities, numOfBodies, dimension);
	}
	cudaThreadSynchronize();
}

__global__ void UpdateValues_Cuda(nbody* d_allBodies, int numOfBodies) {

	int j = (blockIdx.x * blockDim.x) + threadIdx.x;

	float totalForceX = 0;  // total forces acted on x-component of each body
	float totalForceY = 0;  // total forces acted on y-component of each body
	float bodyAccelX = 0;   // calculated x-component acceleration of each body
	float bodyAccelY = 0;   // calculated y-component acceleration of each body

	for (int k = 0; k < numOfBodies; k++) {
		if (k != j) {
			float vec_x = d_allBodies[k].x - d_allBodies[j].x;		// x-position vector
			float vec_y = d_allBodies[k].y - d_allBodies[j].y;		// y-position vector
			float vec_mag = sqrt(vec_x * vec_x + vec_y * vec_y);    // vector magnitude
			float denominator = (vec_mag * vec_mag + SOFTENING * SOFTENING);

			// calculate force components
			float x_comp = (d_allBodies[k].m * vec_x) / sqrt(denominator * denominator * denominator);
			float y_comp = (d_allBodies[k].m * vec_y) / sqrt(denominator * denominator * denominator);

			// update force components
			totalForceX += x_comp;
			totalForceY += y_comp;
		}
	}

	// calculate acceleration components by multiplying the total forces by G
	bodyAccelX = totalForceX * G;
	bodyAccelY = totalForceY * G;

	// calculate velocity components
	d_allBodies[j].vx += dt * bodyAccelX;
	d_allBodies[j].vy += dt * bodyAccelY;

	// calculate new body position components
	d_allBodies[j].x += dt * d_allBodies[j].vx;
	d_allBodies[j].y += dt * d_allBodies[j].vy;
}


__global__ void CountHeatDensitiesCuda(nbody* d_allBodies, float* d_heatDensities, int numOfBodies, int dimension) {
	// calculate and update body densities on each grid during simulation
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	float x_normalised = floor(dimension * d_allBodies[i].x);
	float y_normalised = floor(dimension * d_allBodies[i].y);

	// get the xy_normalised value from the x and y values above
	int xy_normalised = x_normalised + (y_normalised * dimension);  // multiply y_normalised by D as y is the row value

	// make sure we ignore bodies that have gone out of range of the grid
	if ((xy_normalised >= 0) && (xy_normalised < dimension * dimension) && (i < numOfBodies)) {
		atomicAdd(&d_heatDensities[xy_normalised], ((float)dimension / numOfBodies));
	}
}


void step(void)
{
	// Perform the main simulation of the NBody system per iteration
	int j, k;
#pragma omp parallel for private(j, k) schedule(dynamic) if (op_mode == OPENMP)
	for (j = 0; j < numOfBodies; j++) {
		float totalForceX = 0;  // total forces acted on x-component of each body
		float totalForceY = 0;  // total forces acted on y-component of each body
		float bodyAccelX = 0;    // calculated x-component acceleration of each body
		float bodyAccelY = 0;    // calculated y-component acceleration of each body

		for (int k = 0; k < numOfBodies; k++) {
			if (k != j) {
				totalForceX += SumForces(allBodies[j], allBodies[k]).x;
				totalForceY += SumForces(allBodies[j], allBodies[k]).y;
			}
		}

		// calculate acceleration components by multiplying the total forces by G
		bodyAccelX = totalForceX * G;
		bodyAccelY = totalForceY * G;

		// calculate velocity components
		allBodies[j].vx += dt * bodyAccelX;
		allBodies[j].vy += dt * bodyAccelY;

		// calculate new body position components
		allBodies[j].x += dt * allBodies[j].vx;
		allBodies[j].y += dt * allBodies[j].vy;
		//PrintBodyInfo(allBodies[j]);
	}

	// make heatmap calculations if using visualiser mode
	if (iterations == 0) {
		CountHeatDensities();
	}
}

// Calculation of force on a body i by a body j
force SumForces(nbody i, nbody j) {

	float vec_x = j.x - i.x;						        // x-position vector
	float vec_y = j.y - i.y;						        // y-position vector
	float vec_mag = sqrt(vec_x * vec_x + vec_y * vec_y);    // vector magnitude
	float denominator = (vec_mag * vec_mag + SOFTENING * SOFTENING);
	// calculate force components
	float x_comp = (j.m * vec_x) / sqrt(denominator * denominator * denominator);
	float y_comp = (j.m * vec_y) / sqrt(denominator * denominator * denominator);

	// store force components in structure which is returned
	struct force force;
	force.x = x_comp;
	force.y = y_comp;

	return force;

}

void CountHeatDensities(void) {
	// calculate and update body densities on each grid during simulation
	int i;
	#pragma omp parallel for private(i) shared(heatDensities) if (op_mode == OPENMP)
	for (i = 0; i < numOfBodies; i++) {
		float x_normalised = floor(dimension * allBodies[i].x);
		float y_normalised = floor(dimension * allBodies[i].y);
		// get the xy_normalised value from the x and y values above
		int xy_normalised = x_normalised + (y_normalised * dimension);  // multiply y_normalised by D as y is the row value

																		// make sure we ignore bodies that have gone out of range of the grid
		if ((xy_normalised >= 0) && (xy_normalised < dimension * dimension)) {
			#pragma omp atomic
			heatDensities[xy_normalised]++;
		}
	}

	int x;
	#pragma omp parallel for private(x) shared(heatDensities) if (op_mode == OPENMP)
	for (x = 0; x < (dimension * dimension); x++) {
		#pragma omp atomic
		heatDensities[x] /= numOfBodies;
	}
}

void print_help() {
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);
	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}