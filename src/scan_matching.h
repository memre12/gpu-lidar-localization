#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace ScanMatching {
	// Upload/replace the target (map) point cloud. Persistent across frames.
	void setTarget(int M, glm::vec3* target_pc);

	// Upload the source scan and reset the accumulated transform.
	// Device buffers are reused (grow-only) — cheap to call every frame.
	void setSource(int N, glm::vec3* src_pc);

	// Convenience wrapper: setTarget + setSource.
	void initSimulation(int N1, glm::vec3* src_pc, int N2, glm::vec3* target_pc);

	// One ICP iteration. Returns this iteration's motion (|t| + rotation angle).
	// max_corr_dist > 0 gates correspondences: pairs farther apart than this
	// are excluded from the transform estimate (outlier rejection).
	float transformGPU(float dt, float max_corr_dist = 0.0f);
	void transformCPU(float dt);
	void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

	// Accumulated ICP transform since the last setSource.
	// R is row-major 3x3, t is xyz. Maps initial source points to aligned points.
	void getTransform(float R[9], float t[3]);

	void endSimulation();
}