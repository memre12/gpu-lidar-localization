#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "scan_matching.h"
#include "svd3.h"
#include "kdtree.h"

#define KD_TREE 0
#define NAIVE 1

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define NULL_VEC -7879


void printArray2D(glm::mat3& X) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			printf("%f ", X[j][i]);
		printf("\n");
	}
	printf("\n");
}

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

struct TreeNode{
	int pointIdx;
	int parentIdx;
	bool good;
	int depth;
};


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1



glm::vec3 *dev_src_pc = nullptr;
glm::vec3 *dev_src_pc_shift = nullptr;
glm::vec3 *dev_target_pc = nullptr;

glm::vec3 *dev_kdTreeTarget = nullptr;
TreeNode *host_stack = nullptr;
TreeNode *dev_stack = nullptr;

glm::vec3 *dev_X_mean_sub = nullptr;
glm::vec3 *dev_Y_mean_sub = nullptr;

glm::vec3* dev_corres = nullptr;

// Per-pair gate weight (1 = valid correspondence, 0 = rejected outlier)
float* dev_w = nullptr;

glm::mat3* dev_M_yxt = nullptr;

int N_SRC;
int N_TARGET;

// Allocated capacities (grow-only, buffers reused across frames)
int SRC_CAP = 0;
int TARGET_CAP = 0;

int STACK_SIZE;

// Accumulated ICP transform across iterations (since setSource)
glm::mat3 g_icp_R(1.0f);
glm::vec3 g_icp_t(0.0f);


/******************
* initSimulation *
******************/
__global__ void kernResetBuffer(int N, glm::vec3 *buffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		buffer[index] = value;
	}
}

/**
* Upload/replace the target (map) cloud. Stays on the GPU across frames.
*/
void ScanMatching::setTarget(int M, glm::vec3* target_pc) {
	if (M > TARGET_CAP) {
		cudaFree(dev_target_pc);
		cudaMalloc((void**)&dev_target_pc, M * sizeof(glm::vec3));
		checkCUDAErrorWithLine("cudaMalloc target failed!");
		TARGET_CAP = M;
	}
	cudaMemcpy(dev_target_pc, target_pc, M * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	N_TARGET = M;

#if KD_TREE
	STACK_SIZE =  (int)ceil(log2(N_TARGET)) + 1;
	printf("STACK SIZE %d\n", STACK_SIZE);

	cudaFree(dev_stack);
	cudaMalloc((void **)&dev_stack, SRC_CAP * STACK_SIZE * sizeof(TreeNode));
	checkCUDAErrorWithLine("cudaMalloc failed!");

	int treeSize = 1 << ((int)ceil(log2(N_TARGET)) + 1);
	glm::vec3* kdTreeTarget = new glm::vec3[treeSize];
	constructKDTree(target_pc, N_TARGET, kdTreeTarget, treeSize);

	std::cout << "KD Tree Built sucessfully" << '\n';
	cudaFree(dev_kdTreeTarget);
	cudaMalloc((void **)&dev_kdTreeTarget, treeSize * sizeof(glm::vec3));
	cudaMemcpy(dev_kdTreeTarget, kdTreeTarget, treeSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMalloc failed!");
	delete[] kdTreeTarget;
#endif
}

/**
* Upload the source scan; device work buffers are reused across frames.
*/
void ScanMatching::setSource(int N, glm::vec3* src_pc) {
	if (N > SRC_CAP) {
		cudaFree(dev_src_pc);
		cudaFree(dev_src_pc_shift);
		cudaFree(dev_X_mean_sub);
		cudaFree(dev_Y_mean_sub);
		cudaFree(dev_corres);
		cudaFree(dev_w);
		cudaFree(dev_M_yxt);

		cudaMalloc((void**)&dev_src_pc, N * sizeof(glm::vec3));
		cudaMalloc((void**)&dev_src_pc_shift, N * sizeof(glm::vec3));
		cudaMalloc((void**)&dev_X_mean_sub, N * sizeof(glm::vec3));
		cudaMalloc((void**)&dev_Y_mean_sub, N * sizeof(glm::vec3));
		cudaMalloc((void **)&dev_corres, N * sizeof(glm::vec3));
		cudaMalloc((void **)&dev_w, N * sizeof(float));
		cudaMalloc((void **)&dev_M_yxt, N * sizeof(glm::mat3));
		checkCUDAErrorWithLine("cudaMalloc source failed!");
		SRC_CAP = N;
	}
	cudaMemcpy(dev_src_pc, src_pc, N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	N_SRC = N;

	g_icp_R = glm::mat3(1.0f);
	g_icp_t = glm::vec3(0.0f);
}

/**
* Legacy one-shot init: upload target and source together.
*/
void ScanMatching::initSimulation(int N, glm::vec3* src_pc, int M, glm::vec3* target_pc) {
	setTarget(M, target_pc);
	setSource(N, src_pc);
	cudaDeviceSynchronize();
}

/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N_SRC, glm::vec3 *pos_src, int N_TARGET, glm::vec3 *pos_target,  float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N_SRC + N_TARGET) {
		if (index < N_SRC) {
			vbo[4 * index + 0] = pos_src[index].x * c_scale;
			vbo[4 * index + 1] = pos_src[index].y * c_scale;
			vbo[4 * index + 2] = pos_src[index].z * c_scale;
			vbo[4 * index + 3] = 1.0f;
		}
		else {
			vbo[4 * index + 0] = pos_target[index - N_SRC].x * c_scale;
			vbo[4 * index + 1] = pos_target[index - N_SRC].y * c_scale;
			vbo[4 * index + 2] = pos_target[index - N_SRC].z * c_scale;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}

__global__ void kernCopyVelocitiesToVBO(int N_SRC, int N_TARGET, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N_SRC + N_TARGET) {
		if (index < N_SRC) {
			vbo[4 * index + 0] = 1.0 + 0.3f;
			vbo[4 * index + 1] = 1.0 + 0.3f;
			vbo[4 * index + 2] = 1.0 + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
		else {
			vbo[4 * index + 0] = 1.0 + 0.3f;
			vbo[4 * index + 1] = 0.0 + 0.3f;
			vbo[4 * index + 2] = 0.0 + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}


/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void ScanMatching::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((N_SRC + N_TARGET + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (N_SRC, dev_src_pc, N_TARGET, dev_target_pc, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (N_SRC, N_TARGET, vbodptr_velocities, scene_scale);
	checkCUDAErrorWithLine("copyBoidsToVBO failed!");
	cudaDeviceSynchronize();
}


void findCorrespondences(int numSrc, glm::vec3* dev_src, int numTarget, glm::vec3* dev_target, glm::vec3* dev_corres) {

	for (int idx = 0; idx < numSrc; idx++) {
		glm::vec3 me = dev_src[idx];
		int minIdx = 0;
		float minDistance = glm::distance(me, dev_target[0]);
		for (int i = 1; i < numTarget; i++) {
			float d = glm::distance(me, dev_target[i]);
			if (d < minDistance) {
				minIdx = i;
				minDistance = d;
			}
		}

		dev_corres[idx] = dev_target[minIdx];
	}
}

__global__ void kernFindCorrespondences(int numSrc, glm::vec3* dev_src, int numTarget, glm::vec3* dev_target, glm::vec3* dev_corres, float* dev_w, float maxDist2) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	if (idx >= numSrc) return;

	glm::vec3 me = dev_src[idx];
	int minIdx = 0;
	glm::vec3 diff = me - dev_target[0];
	float minDistance = glm::dot(diff, diff);  // squared distance: no sqrt needed for argmin
	for (int i = 1; i < numTarget; i++) {
		diff = me - dev_target[i];
		float d = glm::dot(diff, diff);
		if ( d < minDistance) {
			minIdx = i;
			minDistance = d;
		}
	}

	dev_corres[idx] = dev_target[minIdx];
	// Gate: pairs farther apart than maxDist2 don't contribute to the estimate
	dev_w[idx] = (maxDist2 <= 0.0f || minDistance < maxDist2) ? 1.0f : 0.0f;
}

__device__ TreeNode createNode(int pointIdx, int parentIdx, bool good, int depth) {
	TreeNode node;
	node.pointIdx = pointIdx;
	node.parentIdx = parentIdx;
	node.good = good;
	node.depth = depth;
	return node; 
}

__global__ void kernFindCorrespondencesKDTree(
	int numSrc, glm::vec3* dev_src, 
	int numTarget, glm::vec3* kdTreeTarget, 
	glm::vec3* dev_corres, 
	TreeNode* stack, int stackSize) {

	int tidx = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tidx >= numSrc) return;

	int bestIdx = 0;
	float  bestDistance = LONG_MAX;

	//push root onto stack
	int root_idx = 0;
	int top = 0;

	stack[tidx*stackSize + top] = createNode(root_idx, -1, true, 0);
	
	glm::vec3 queryPoint = dev_src[tidx];

	while (top != -1) {

		// pop  from stack
		TreeNode curNode = stack[tidx*stackSize + top--];

		int curIdx = curNode.pointIdx;
		glm::vec3 curPoint = kdTreeTarget[curIdx];

		if (curPoint.z == NULL_VEC) continue;

		int curDepth = curNode.depth;
		int axis = curDepth % 3;
		bool good = curNode.good;

		if (!good) {
			// check if bad side should be searched
			int parentIdx = curNode.parentIdx;
			glm::vec3 parentPoint = kdTreeTarget[parentIdx];
			int parentAxis = curNode.depth % 3;
			if (abs(parentPoint[parentAxis] - queryPoint[parentAxis]) > bestDistance) continue;
		}

		float curDistance = glm::distance(curPoint, queryPoint);
		if ( curDistance < bestDistance) {
			bestIdx = curIdx;
			bestDistance = curDistance;
		}

		int goodIdx, badIdx;
		if (queryPoint[axis] < curPoint[axis]) {
			goodIdx = 2 * curIdx + 1; // search left child first
			badIdx = 2 * curIdx + 2; // search right child second
		}
		else {
			goodIdx = 2 * curIdx + 2; // search right child first
			badIdx = 2 * curIdx + 1; // search left child second
		}

		// push bad node first and good node later
		TreeNode goodNode = createNode(goodIdx, curIdx, true, curDepth + 1);
		TreeNode badNode = createNode(badIdx, curIdx, false, curDepth + 1);
		
		stack[tidx*stackSize + ++top] = badNode;
		stack[tidx*stackSize + ++top] = goodNode;
	}

	dev_corres[tidx] = kdTreeTarget[bestIdx];
}

/*
	Calculate Y (target 3 x N) x X.T (source N x 3) result 3 x 3 
*/

__global__ void kernMultiplyYXTranspose(int N, glm::vec3* tars, glm::vec3* srcs, glm::mat3* outM) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	if (idx >= N) return;

	glm::vec3 s = srcs[idx];
	glm::vec3 t = tars[idx];

	outM[idx] = glm::outerProduct(t, s);

	//outM[idx] = glm::mat3(t.x*s.x, t.x*s.y, t.x*s.z,
	//					t.y*s.x, t.y*s.y, t.y*s.z,
	//					t.z*s.x, t.z*s.y, t.z*s.z);

}

struct mean_center_op
{
	const glm::vec3 mean;

	mean_center_op(glm::vec3 _a) : mean(_a) {}

	__host__ __device__
		glm::vec3 operator()(const glm::vec3& x) const {
		return x - mean;
	}
};

struct transform_src_op
{
	const glm::vec3 t;
	const glm::mat3 R;

	transform_src_op(glm::vec3 _a, glm::mat3 _b) : t(_a), R(_b) {}

	__host__ __device__
		glm::vec3 operator()(const glm::vec3& x) const {
		return R*x + t;
	}
};


void ScanMatching::transformCPU(float dt) {


}


// Multiply each point by its gate weight (invalid pairs contribute zero)
struct weight_op {
	__host__ __device__
	glm::vec3 operator()(const glm::vec3& x, const float& w) const {
		return x * w;
	}
};

// w * (x - mean): invalid pairs drop out of the covariance sum
struct weighted_center_op {
	const glm::vec3 mean;
	weighted_center_op(glm::vec3 _m) : mean(_m) {}

	__host__ __device__
	glm::vec3 operator()(const glm::vec3& x, const float& w) const {
		return (x - mean) * w;
	}
};

float ScanMatching::transformGPU(float dt, float max_corr_dist) {
	dim3 numBlocksPerGrid((N_SRC + blockSize - 1) / blockSize);
	float maxDist2 = max_corr_dist > 0.0f ? max_corr_dist * max_corr_dist : 0.0f;

#if KD_TREE
	kernFindCorrespondencesKDTree <<< numBlocksPerGrid, blockSize >>> (N_SRC, dev_src_pc, N_TARGET, dev_kdTreeTarget, dev_corres, dev_stack, STACK_SIZE);
#endif

#if NAIVE
	kernFindCorrespondences << < numBlocksPerGrid, blockSize >> > (N_SRC, dev_src_pc, N_TARGET, dev_target_pc, dev_corres, dev_w, maxDist2);
#endif
	float sum_w = thrust::reduce(thrust::device, dev_w, dev_w + N_SRC, 0.0f);
	if (sum_w < 50.0f) {
		// Too few valid correspondences to estimate a transform — skip
		return 0.0f;
	}

	// Weighted means over valid pairs only
	thrust::transform(thrust::device, dev_src_pc, dev_src_pc + N_SRC, dev_w, dev_X_mean_sub, weight_op());
	glm::vec3 mean_src = thrust::reduce(thrust::device, dev_X_mean_sub, dev_X_mean_sub + N_SRC, glm::vec3(0.0f));
	mean_src /= sum_w;

	thrust::transform(thrust::device, dev_corres, dev_corres + N_SRC, dev_w, dev_Y_mean_sub, weight_op());
	glm::vec3 mean_tar = thrust::reduce(thrust::device, dev_Y_mean_sub, dev_Y_mean_sub + N_SRC, glm::vec3(0.0f));
	mean_tar /= sum_w;

	// Mean-center; the source side carries the weight so gated pairs
	// contribute zero to the covariance
	thrust::transform(thrust::device, dev_src_pc, dev_src_pc + N_SRC, dev_w, dev_X_mean_sub, weighted_center_op(mean_src));
	thrust::transform(thrust::device, dev_corres, dev_corres + N_SRC, dev_Y_mean_sub, mean_center_op(mean_tar));

	kernMultiplyYXTranspose<<< numBlocksPerGrid, blockSize >>> (N_SRC, dev_Y_mean_sub, dev_X_mean_sub, dev_M_yxt);

	glm::mat3 host_M = thrust::reduce(thrust::device, dev_M_yxt, dev_M_yxt + N_SRC, glm::mat3(0.0f));

	glm::mat3 U(0.0f);
	glm::mat3 S(0.0f);
	glm::mat3 V(0.0f);

	svd(host_M[0][0], host_M[1][0], host_M[2][0],
		host_M[0][1], host_M[1][1], host_M[2][1],
		host_M[0][2], host_M[1][2], host_M[2][2],
		U[0][0], U[1][0], U[2][0],
		U[0][1], U[1][1], U[2][1],
		U[0][2], U[1][2], U[2][2],
		S[0][0], S[1][0], S[2][0],
		S[0][1], S[1][1], S[2][1],
		S[0][2], S[1][2], S[2][2],
		V[0][0], V[1][0], V[2][0],
		V[0][1], V[1][1], V[2][1],
		V[0][2], V[1][2], V[2][2]);


	glm::mat3 R(0.0f);
	R = U * glm::transpose(V);

	glm::vec3 t = mean_tar - R * mean_src;


	thrust::transform(thrust::device, dev_src_pc, dev_src_pc + N_SRC, dev_src_pc_shift, transform_src_op(t, R));

	glm::vec3* tmp = dev_src_pc;
	dev_src_pc = dev_src_pc_shift;
	dev_src_pc_shift =tmp;

	// No explicit sync needed: the thrust reductions above already
	// synchronized, and remaining work is stream-ordered.

	// Accumulate this iteration's transform: p' = R*p + t
	g_icp_R = R * g_icp_R;
	g_icp_t = R * g_icp_t + t;

	// Convergence metric: translation magnitude + rotation angle of this iteration
	float trace = R[0][0] + R[1][1] + R[2][2];
	float cos_angle = (trace - 1.0f) * 0.5f;
	cos_angle = fminf(1.0f, fmaxf(-1.0f, cos_angle));
	float angle = acosf(cos_angle);

	return glm::length(t) + angle;
}

void ScanMatching::getTransform(float R[9], float t[3]) {
	// glm::mat3 is column-major: g_icp_R[col][row] -> row-major output
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			R[i * 3 + j] = g_icp_R[j][i];
		}
		t[i] = g_icp_t[i];
	}
}

void ScanMatching::endSimulation() {

	cudaFree(dev_src_pc);
	cudaFree(dev_src_pc_shift);
	cudaFree(dev_target_pc);
	cudaFree(dev_X_mean_sub);
	cudaFree(dev_Y_mean_sub);
	cudaFree(dev_corres);
	cudaFree(dev_w);
	cudaFree(dev_M_yxt);

	dev_src_pc = nullptr;
	dev_src_pc_shift = nullptr;
	dev_target_pc = nullptr;
	dev_X_mean_sub = nullptr;
	dev_Y_mean_sub = nullptr;
	dev_corres = nullptr;
	dev_w = nullptr;
	dev_M_yxt = nullptr;

	SRC_CAP = 0;
	TARGET_CAP = 0;

#if KD_TREE
	cudaFree(dev_kdTreeTarget);
	cudaFree(dev_stack);
	dev_kdTreeTarget = nullptr;
	dev_stack = nullptr;
#endif
}