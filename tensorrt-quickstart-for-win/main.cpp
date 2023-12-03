#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;

class SampleSegmentation
{

public:
    SampleSegmentation(const std::string& engineFilename);
    bool infer(const std::string& input_filename, int32_t width, int32_t height, const std::string& output_filename);

private:
    std::string mEngineFilename;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    util::UniquePtr<nvinfer1::ICudaEngine> mEngine;
};

SampleSegmentation::SampleSegmentation(const std::string& engineFilename)
    : mEngineFilename(engineFilename)
    , mEngine(nullptr)
{
    std::ifstream engineFile(engineFilename, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);
}

bool SampleSegmentation::infer(const std::string& input_filename, int32_t width, int32_t height, const std::string& output_filename)
{
    auto context = util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    auto input_idx = mEngine->getBindingIndex("input");
    if (input_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, 3, height, width};
    context->setBindingDimensions(input_idx, input_dims);
    auto input_size = util::getMemorySize(input_dims, sizeof(float));

    auto output_idx = mEngine->getBindingIndex("output");
    if (output_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(output_idx) == nvinfer1::DataType::kINT32);
    auto output_dims = context->getBindingDimensions(output_idx);
    auto output_size = util::getMemorySize(output_dims, sizeof(int32_t));

    void* input_mem{nullptr};
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void* output_mem{nullptr};
    if (cudaMalloc(&output_mem, output_size) != cudaSuccess)
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }

    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)};
    input_image.read();
    auto input_buffer = input_image.process();
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    if (cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }

    void* bindings[] = {input_mem, output_mem};
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    auto output_buffer = std::unique_ptr<int>{new int[output_size]};
    if (cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    const int num_classes{21};
    const std::vector<int> palette{(0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 << 21) - 1};
    auto output_image{util::ArgmaxImageWriter(output_filename, output_dims, palette, num_classes)};
    output_image.process(output_buffer.get());
    output_image.write();

    cudaFree(input_mem);
    cudaFree(output_mem);
    return true;
}

int main(int argc, char** argv)
{
    clock_t start = clock();

    int32_t width{1282};
    int32_t height{1026};

    SampleSegmentation sample("fcn-resnet101.engine");

    gLogInfo << "Running TensorRT inference for FCN-ResNet101" << std::endl;
    if (!sample.infer("input.ppm", width, height, "output.ppm"))
    {
        return -1;
    }

    clock_t end = clock();

    const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("time %lf[ms]\n", time);

    return 0;
}
