#pragma once

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>
#include <vector>


class onnxModel
{
public:
    onnxModel(const wchar_t* onnx_model_path);
    std::vector<float> predict(std::vector<float>& input_data, int batch_size = 1, int index = 0);
    cv::Mat predict(cv::Mat& input_tensor, int batch_size = 1, int index = 0);
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*>input_node_names;
    std::vector<const char*>output_node_names;
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
};


#ifdef __cplusplus
extern "C" {
#endif

typedef struct onnxModel OnnxModel;

// Model constructor
OnnxModel* OnnxModel_new(const char* onnx_model_path);

// Inference function
unsigned char* OnnxModel_predict(OnnxModel* model, const char* data, int size, int* out_size);

// Model destructor
void OnnxModel_delete(OnnxModel* model);

// Free malloc
void free_malloc(void *ptr);

#ifdef __cplusplus
}
#endif