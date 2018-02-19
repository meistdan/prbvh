/**
 * \file	PRBVHBuilder.h
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	PRBVHBuilder class header file.
 */

#ifndef _PRBVH_BUILDER_H_
#define _PRBVH_BUILDER_H_

#include "LBVHBuilder.h"
#include "PRBVHBuilderKernels.h"

namespace FW {

class PRBVHBuilder : public LBVHBuilder {

private:

	CudaCompiler prbvhCompiler;

	Buffer locks;
	Buffer areaReductions;
	Buffer outNodeIndices;
	Buffer paths;
	Buffer zeroBounds;

	int mod;

	void configure(void);
	void allocate(int numberOfTriangles);

	float optimize(int numberOfTriangles);
	float computeSizes(int numberOfTriangles);
	float build(CudaBVH & bvh, Scene * scene);

public:

	PRBVHBuilder(void);
	virtual ~PRBVHBuilder(void);

	CudaBVH * build(Scene * scene);

	int getMod(void);
	void setMod(int mod);

};

};

#endif /* _PRBVH_BUILDER_H_ */
