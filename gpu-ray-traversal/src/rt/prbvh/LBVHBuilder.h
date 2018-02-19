/**
 * \file	LBVHBuilder.h
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	LBVHBuilder class header file.
 */

#ifndef _LBVH_BULDER_H_
#define _LBVH_BULDER_H_

#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaBVH.hpp"
#include "gpu/Buffer.hpp"

#include "BVHCollapser.h"
#include "LBVHBuilderKernels.h"

namespace FW {

#define VALIDATE_BVH 1

class LBVHBuilder {

protected:

	CudaCompiler compiler;
	BVHCollapser collapser;

	Buffer triangleIndices;
	Buffer mortonCodes[2];

	Buffer nodeLeftIndices;
	Buffer nodeRightIndices;
	Buffer nodeSizes;
	Buffer nodeBoxesMin;
	Buffer nodeBoxesMax;
	Buffer nodeParentIndices;

	Buffer termCounters;

	bool adaptiveLeafSize;
	bool mortonCodes60Bits;
	int maxLeafSize;

	bool validate(CudaBVH & bvh, Scene * scene);

	void configure(void);
	void allocate(int numberOfTriangles);

	float computeSceneBox(Scene * scene);
	float computeMortonCodes(Scene * scene);
	float sortTriangles(int numberOfTriangles, CudaBVH & bvh);
	float setupLeaves(int numberOfTriangles);
	float construct(int numberOfTriangles);
	float refit(int numberOfTriangles);
	float woopifyTriangles(CudaBVH & bvh, Scene * scene);
	float computeCost(CudaBVH & bvh);
	virtual float build(CudaBVH & bvh, Scene * scene);

public:

	LBVHBuilder(void);
	virtual ~LBVHBuilder(void);

	virtual CudaBVH * build(Scene * scene);

	bool isMortonCodes60Bits(void);
	void setMortonCodes60Bits(bool mortonCodes60Bits);
	bool getAdaptiveLeafSize(void);
	void setAdaptiveLeafSize(bool adaptiveLeafSize);
	int getMaxLeafSize(void);
	void setMaxLeafSize(int maxLeafSize);

};

};
#endif /* _LBVH_BULDER_H_ */
