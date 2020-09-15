//Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

#define G			9.8f		//gravitational constant
#define dt			0.01f		//time step
#define SOFTENING	2.0f		//softening parameter to help with numerical instability

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct nbody{
	float x, y, vx, vy, m;
};

struct nbody_soa {
	float* x, * y, * vx, * vy, * m;
};

// struct for seperating force into x and y components
struct force {
	float x, y;
};

typedef enum MODE { CPU, OPENMP, CUDA } MODE;
typedef struct nbody nbody;
typedef struct nbody_soa nbody_soa;
typedef struct force force;
force SumForces(nbody i, nbody j);
void CountHeatDensities();
void step_cuda();

__global__ void CountHeatDensitiesCuda(nbody* b, float* h, int n, int d);
__global__ void UpdateValues_Cuda(nbody* b, int n);

#endif	//__NBODY_HEADER__---=
