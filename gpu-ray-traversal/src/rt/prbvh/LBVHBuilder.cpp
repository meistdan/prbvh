/**
 * \file	LBVHBuilder.cpp
 * \author	Daniel Meister
 * \date	2017/01/22
 * \brief	LBVHBuilder class source file.
 */

#include "LBVHBuilder.h"
#include "BVHUtil.h"
#include "Cub.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>

using namespace FW;

bool LBVHBuilder::validate(CudaBVH & bvh, Scene * scene) {

	bool valid = true;

	// Nodes.
	CudaBVHNode * nodes = (CudaBVHNode*)bvh.getNodeBuffer().getPtr();

	// Triangle indices.
	int * triangleIndices = (int*)bvh.getTriIndexBuffer().getPtr();

	// Nodes histogram.
	int numberOfNodes = int(bvh.getNodeBuffer().getSize() / sizeof(CudaBVHNode));
	std::vector<int> nodeHistogram(numberOfNodes);
	memset(nodeHistogram.data(), 0, sizeof(int)* numberOfNodes);
	nodeHistogram[0]++;

	// Triangle histogram.
	int numberOfTriangles = scene->getNumTriangles();
	std::vector<int> triangleHistogram(numberOfTriangles);
	memset(triangleHistogram.data(), 0, sizeof(int)* numberOfTriangles);

	// Check triangle indices.
	for (int i = 0; i < numberOfTriangles; ++i) {
		triangleHistogram[triangleIndices[i]]++;
	}

	for (int i = 0; i < numberOfTriangles; ++i) {
		if (triangleHistogram[i] < 1) {
			printf("<LBVHBuilder> Invalid triangle indices!\n");
			valid = false;
		}
	}

	// Reset triangle histogram.
	memset(triangleHistogram.data(), 0, sizeof(int)* numberOfTriangles);

	// Stack.
	std::stack<int> stack;
	stack.push(0);

	// Traverse BVH.
	while (!stack.empty()) {

		// Pop.
		int nodeIndex = stack.top();
		stack.pop();
		CudaBVHNode & node = nodes[nodeIndex];

		// Interior.
		if (!node.isLeaf()) {

			// Child indices.
			int leftIndex = node.begin < 0 ? ~node.begin : node.begin;
			int rightIndex = node.end < 0 ? ~node.end : node.end;

			// Child nodes.
			CudaBVHNode & left = nodes[leftIndex];
			CudaBVHNode & right = nodes[rightIndex];

			// Parent index.
			if (left.getParentIndex() != nodeIndex || right.getParentIndex() != nodeIndex) {
				printf("<LBVHBuilder> Invalid parent index!\n");
				valid = false;
			}

			// Check sizes.
			if (node.getSize() != left.getSize() + right.getSize()) {
				printf("<LBVHBuilder> Invalid node size!\n");
				valid = false;
			}

			// Update histogram.
			nodeHistogram[leftIndex]++;
			nodeHistogram[rightIndex]++;

			// Push.
			stack.push(leftIndex);
			stack.push(rightIndex);

		}

		// Leaf.
		else {

			// Check Bounds.
			if (node.begin >= node.end) {
				printf("<LBVHBuilder> Invalid leaf bounds [%d, %d]!\n", node.begin, node.end);
				valid = false;
			}

			// Update histogram.
			for (int i = node.begin; i < node.end; ++i) {
				int triangleIndex = triangleIndices[i];
				triangleHistogram[triangleIndex]++;
			}

		}

	}

	// Check node histogram.
	for (int i = 0; i < numberOfNodes; ++i) {
		if (nodeHistogram[i] != 1) {
			printf("<LBVHBuilder> Not all nodes are referenced!\n");
			valid = false;
		}
	}

	// Check triangle histogram.
	for (int i = 0; i < numberOfTriangles; ++i) {
		if (triangleHistogram[i] < 1) {
			printf("<LBVHBuilder> Not all triangles are referenced!\n");
			valid = false;
		}
	}

	return valid;

}

void LBVHBuilder::configure() {

	const std::string configFile = "prbvh.cfg";

	std::ifstream file(configFile);
	std::string line, key, value;

	int maxLeafSizeTmp;

	if (file.is_open()) {
		while (getline(file, line)) {
			std::istringstream ss(line);
			getline(ss, key, '=');
			getline(ss, value, '=');

			if (key == "morton60") {
				std::istringstream(value) >> mortonCodes60Bits;
			}
			else if (key == "adaptiveLeafSize") {
				std::istringstream(value) >> adaptiveLeafSize;
			}
			else if (key == "maxLeafSize") {
				std::istringstream(value) >> maxLeafSizeTmp;
				setMaxLeafSize(maxLeafSizeTmp);
			}

		}
		file.close();
	}

}

void LBVHBuilder::allocate(int numberOfTriangles) {
	if (mortonCodes60Bits) {
		mortonCodes[0].resizeDiscard(sizeof(unsigned long long) * numberOfTriangles);
		mortonCodes[1].resizeDiscard(sizeof(unsigned long long) * numberOfTriangles);
	}
	else {
		mortonCodes[0].resizeDiscard(sizeof(unsigned int) * numberOfTriangles);
		mortonCodes[1].resizeDiscard(sizeof(unsigned int) * numberOfTriangles);
	}
	nodeLeftIndices.resizeDiscard(sizeof(int) * (2 * numberOfTriangles - 1));
	nodeRightIndices.resizeDiscard(sizeof(int) * (2 * numberOfTriangles - 1));
	nodeSizes.resizeDiscard(sizeof(int) * (2 * numberOfTriangles - 1));
	nodeBoxesMin.resizeDiscard(sizeof(Vec4f) * (2 * numberOfTriangles - 1));
	nodeBoxesMax.resizeDiscard(sizeof(Vec4f) * (2 * numberOfTriangles - 1));
	nodeParentIndices.resizeDiscard(sizeof(int) * (2 * numberOfTriangles - 1));
	triangleIndices.resizeDiscard(sizeof(int) * numberOfTriangles);
	termCounters.resizeDiscard(sizeof(int) * numberOfTriangles);
}

float LBVHBuilder::computeSceneBox(Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeSceneBox");

	// Texture memory.
	module->setTexRef("vertexTex", scene->getVtxPosBuffer(), CU_AD_FORMAT_FLOAT, 1);

	// Scene box.
	*(AABB*)module->getGlobal("sceneBox").getMutablePtr() = AABB();

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = scene->getNumVertices();

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumVertices()
		);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float LBVHBuilder::computeMortonCodes(Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel(!mortonCodes60Bits ? "computeMortonCodes30" : "computeMortonCodes60");

	// Texture memory.fcon
	module->setTexRef("triangleTex", scene->getTriVtxIndexBuffer(), CU_AD_FORMAT_UNSIGNED_INT32, 1);
	module->setTexRef("vertexTex", scene->getVtxPosBuffer(), CU_AD_FORMAT_FLOAT, 1);

	// Scene box.
	AABB sceneBox = *(AABB*)module->getGlobal("sceneBox").getPtr();
	Vec3f diag = sceneBox.max() - sceneBox.min();
	float edge = max(max(diag.x, diag.y), diag.z);
	sceneBox.max() = sceneBox.min() + Vec3f(edge);
	*(AABB*)module->getGlobal("sceneBoxConst").getMutablePtr() = sceneBox;

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = scene->getNumTriangles();

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumTriangles(),
		mortonCodes[0],
		triangleIndices
		);

	// Launch.
	return kernel.launchTimed(threads, Vec2i(blockThreads, 1));

}

float LBVHBuilder::sortTriangles(int numberOfTriangles, CudaBVH & bvh) {
	float time = 0.0f;
	bool sortSwap = false;
	int * values0 = (int*)triangleIndices.getMutableCudaPtr();
	int * values1 = (int*)bvh.getTriIndexBuffer().getMutableCudaPtr();
	if (mortonCodes60Bits) {
		unsigned long long * keys0 = (unsigned long long*)mortonCodes[0].getMutableCudaPtr();
		unsigned long long * keys1 = (unsigned long long*)mortonCodes[1].getMutableCudaPtr();
		time = Cub::sort(numberOfTriangles, keys0, keys1, values0, values1, sortSwap);
	}
	else {
		unsigned int * keys0 = (unsigned int*)mortonCodes[0].getMutableCudaPtr();
		unsigned int * keys1 = (unsigned int*)mortonCodes[1].getMutableCudaPtr();
		time =  Cub::sort(numberOfTriangles, keys0, keys1, values0, values1, sortSwap);
	}
	if (sortSwap) {
		mortonCodes[0] = mortonCodes[1];
		triangleIndices = bvh.getTriIndexBuffer();
	}
	return time;
}

float LBVHBuilder::setupLeaves(int numberOfTriangles) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("setupLeaves");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = numberOfTriangles;

	// Set params.
	kernel.setParams(
		threads,
		numberOfTriangles,
		triangleIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes,
		nodeBoxesMin,
		nodeBoxesMax
	);
	
	// Launch.
	return kernel.launchTimed(threads, Vec2i(blockThreads, 1));

}

float LBVHBuilder::construct(int numberOfTriangles) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel(mortonCodes60Bits ? "construct60" : "construct30");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	
	// Set params.
	kernel.setParams(
		numberOfTriangles,
		nodeParentIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes,
		mortonCodes[0]
	);
	
	// Launch.
	return kernel.launchTimed(numberOfTriangles, Vec2i(blockThreads, 1));

}

float LBVHBuilder::refit(int numberOfTriangles) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("refit");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;

	// Clear termination counters.
	termCounters.clear();

	// Set params.
	kernel.setParams(
		numberOfTriangles,
		2 * numberOfTriangles - 1,
		termCounters,
		nodeParentIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeBoxesMin,
		nodeBoxesMax
	);
	
	// Launch.
	return kernel.launchTimed(numberOfTriangles, Vec2i(blockThreads, 1));

}

float LBVHBuilder::woopifyTriangles(CudaBVH & bvh, Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("woopifyTriangles");

	// Texture memory.
	module->setTexRef("triangleTex", scene->getTriVtxIndexBuffer(), CU_AD_FORMAT_UNSIGNED_INT32, 1);
	module->setTexRef("vertexTex", scene->getVtxPosBuffer(), CU_AD_FORMAT_FLOAT, 1);

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = scene->getNumTriangles();

	// Woop buffer.
	CUdeviceptr triPtr = bvh.getTriWoopBuffer().getCudaPtr();

	// Woop offsets.
	Vec2i triOfsA = bvh.getTriWoopSubArray(0);
	Vec2i triOfsB = bvh.getTriWoopSubArray(1);
	Vec2i triOfsC = bvh.getTriWoopSubArray(2);

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumTriangles(),
		bvh.getTriIndexBuffer(),
		triPtr + triOfsA.x,
		triPtr + triOfsB.x,
		triPtr + triOfsC.x
		);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float LBVHBuilder::computeCost(CudaBVH & bvh) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeCost");

	// Number of nodes.
	int numberOfNodes = int(bvh.getNodeBuffer().getSize() / sizeof(CudaBVHNode));

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = numberOfNodes;

	// Scee box.
	AABB sceneBox = *(AABB*)module->getGlobal("sceneBox").getPtr();

	// Reset cost.
	*(float*)module->getGlobal("cost").getMutablePtr() = 0.0f;

	// Set params.
	kernel.setParams(
		threads,
		numberOfNodes,
		sceneBox.area(),
		collapser.getCt(),
		collapser.getCi(),
		bvh.getNodeBuffer()
		);

	// Launch.
	kernel.launch(threads, Vec2i(blockThreads, 1));

	// Cost
	return *(float*)module->getGlobal("cost").getPtr();

}

float LBVHBuilder::build(CudaBVH & bvh, Scene * scene) {

	// Number of triangles.
	int numberOfTriangles = scene->getNumTriangles();

	// Allocate buffers.
	allocate(numberOfTriangles);

	// Compute scene box.
	float sceneBoxTime = computeSceneBox(scene);
	printf("<LBVHBuilder> Scene box computed in %fs.\n", sceneBoxTime);

	// Morton codes.
	float mortonCodesTime = computeMortonCodes(scene);
	printf("<LBVHBuilder> Morton codes (%d bits) computed in %fs.\n", mortonCodes60Bits ? 60 : 30, mortonCodesTime);

	// Sort.
	float sortTime = sortTriangles(numberOfTriangles, bvh);
	printf("<LBVHBuilder> Triangles sorted in %fs.\n", sortTime);

	// Setup leaves.
	float setupLeavesTime = setupLeaves(numberOfTriangles);
	printf("<LBVHBuilder> Leaves setup in %fs.\n", setupLeavesTime);

	// Construction.
	float constructTime = construct(numberOfTriangles);
	printf("<LBVHBuilder> Topology constructed in %fs.\n", constructTime);

	// Refit.
	float refitTime = refit(numberOfTriangles);
	printf("<LBVHBuilder> Bounding boxes refitted in %fs.\n", refitTime);

	// Collapse.
	float collapseTime;
	if (adaptiveLeafSize)
		collapseTime = collapser.collapseAdaptive(numberOfTriangles, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, termCounters, bvh);
	else
		collapseTime = collapser.collapse(numberOfTriangles, maxLeafSize, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, termCounters, bvh);
	printf("<LBVHBuilder> BVH collapsed and converted in %fs.\n", collapseTime);

	// Woopify triangles.
	float woopTime = woopifyTriangles(bvh, scene);
	printf("<LBVHBuilder> Triangles woopified in %fs.\n", woopTime);

	return sceneBoxTime + mortonCodesTime + sortTime + setupLeavesTime + constructTime + refitTime + collapseTime + woopTime;

}

LBVHBuilder::LBVHBuilder() : mortonCodes60Bits(true), maxLeafSize(8), adaptiveLeafSize(true) {
	compiler.setSourceFile("src/rt/prbvh/LBVHBuilderKernels.cu");
    compiler.addOptions("-use_fast_math");
	compiler.include("src/rt");
	compiler.include("src/framework");
	configure();
}

LBVHBuilder::~LBVHBuilder() {
}

CudaBVH * LBVHBuilder::build(Scene * scene) {

	// Create BVH.
	CudaBVH * bvh = new CudaBVH(BVHLayout::BVHLayout_AOS_SOA);

	// Resize buffers.
	const int TRIANGLE_ALIGN = 4096;
	bvh->getNodeBuffer().resizeDiscard(sizeof(CudaBVHNode)* (2 * scene->getNumTriangles() - 1));
	bvh->getTriIndexBuffer().resizeDiscard(sizeof(int)* scene->getNumTriangles());
	bvh->getTriWoopBuffer().resizeDiscard((4 * sizeof(Vec4f)* scene->getNumTriangles() + TRIANGLE_ALIGN - 1) & -TRIANGLE_ALIGN);

	// Settings.
	printf("<LBVHBuilder> Morton codes %d bits.\n", mortonCodes60Bits ? 60 : 30);
	printf("<LBVHBuilder> Adaptive collapse %d, Max. leaf size %d.\n", int(adaptiveLeafSize), maxLeafSize);

	// Build.
	float time = build(*bvh, scene);

	// Cost.
	float cost = computeCost(*bvh);

#if VALIDATE_BVH
	// Validate.
	validate(*bvh, scene);
#endif

	// Stats.
	printf("<LBVHBuilder> BVH built in %fs from %d triangle.\n", time, scene->getNumTriangles());
	printf("<LBVHBuilder> %f MTriangles/s.\n", (scene->getNumTriangles() * 1.0e-3f / time), scene->getNumTriangles());
	printf("<LBVHBuilder> BVH cost is %f.\n", cost);

	return bvh;

}

bool LBVHBuilder::isMortonCodes60Bits() {
	return mortonCodes60Bits;
}

void LBVHBuilder::setMortonCodes60Bits(bool mortonCodes60bits) {
	this->mortonCodes60Bits = mortonCodes60bits;
}

bool LBVHBuilder::getAdaptiveLeafSize() {
	return adaptiveLeafSize;
}

void LBVHBuilder::setAdaptiveLeafSize(bool adaptiveLeafSize) {
	this->adaptiveLeafSize = adaptiveLeafSize;
}

int LBVHBuilder::getMaxLeafSize() {
	return maxLeafSize;
}

void LBVHBuilder::setMaxLeafSize(int maxLeafSize) {
	if (maxLeafSize > 0 && maxLeafSize <= 64) this->maxLeafSize = maxLeafSize;
}
