#include "onnxModel.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#define DLOPEN LoadLibraryA
#define DLSYM GetProcAddress
#define DLCLOSE FreeLibrary
#define DLERROR GetLastError
#else
#include <dlfcn.h>
#define DLOPEN dlopen
#define DLSYM dlsym
#define DLCLOSE dlclose
#define DLERROR dlerror
#endif

onnxModel::onnxModel(const wchar_t* onnx_model_path) 
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"), 
      session(nullptr), 
      use_cuda(false),
      cuda_available(false)
{
    // Try to initialize CUDA first, fallback to CPU if not available
    if (!initializeCUDA(onnx_model_path)) {
        std::cout << "CUDA not available, falling back to CPU execution" << std::endl;
        if (!initializeCPU(onnx_model_path)) {
            throw std::runtime_error("Failed to initialize both CUDA and CPU execution providers");
        }
    }
}

onnxModel::~onnxModel() {
    cleanup();
}

bool onnxModel::initializeCUDA(const wchar_t* onnx_model_path) {
    try {
        // Test if CUDA is actually available by trying to create a session with CUDA provider
        Ort::SessionOptions session_options;
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Try to append CUDA execution provider
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;  // kSameAsRequested
        cuda_options.gpu_mem_limit = SIZE_MAX;   // No limit
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        
                 // Create session with CUDA
         session = std::make_unique<Ort::Session>(env, onnx_model_path, session_options);
         
         // Initialize input/output node information
         size_t num_input_nodes = session->GetInputCount();
         size_t num_output_nodes = session->GetOutputCount();
         for (int i = 0; i < num_input_nodes; i++) {
             char* input_node_name = strdup(session->GetInputNameAllocated(i, allocator).get());
             this->input_node_names.push_back(input_node_name);
             Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
             auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
             ONNXTensorElementDataType type = tensor_info.GetElementType();
             this->input_node_dims = tensor_info.GetShape();
         }
         for (int i = 0; i < num_output_nodes; i++) {
             char* output_node_name = strdup(session->GetOutputNameAllocated(i, allocator).get());
             this->output_node_names.push_back(output_node_name);
             Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
             auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
             this->output_node_dims = tensor_info.GetShape();
         }
        
        // If we get here, CUDA is working
        use_cuda = true;
        cuda_available = true;
        
        std::cout << "CUDA execution provider initialized successfully" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cout << "CUDA initialization failed: " << e.what() << std::endl;
        cleanup();
        return false;
    } catch (const std::exception& e) {
        std::cout << "CUDA initialization failed: " << e.what() << std::endl;
        cleanup();
        return false;
    }
    
    return true;
}

bool onnxModel::initializeCPU(const wchar_t* onnx_model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
                 // Create session with CPU only
         session = std::make_unique<Ort::Session>(env, onnx_model_path, session_options);
         
         // Initialize input/output node information
         size_t num_input_nodes = session->GetInputCount();
         size_t num_output_nodes = session->GetOutputCount();
         for (int i = 0; i < num_input_nodes; i++) {
             char* input_node_name = strdup(session->GetInputNameAllocated(i, allocator).get());
             this->input_node_names.push_back(input_node_name);
             Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
             auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
             ONNXTensorElementDataType type = tensor_info.GetElementType();
             this->input_node_dims = tensor_info.GetShape();
         }
         for (int i = 0; i < num_output_nodes; i++) {
             char* output_node_name = strdup(session->GetOutputNameAllocated(i, allocator).get());
             this->output_node_names.push_back(output_node_name);
             Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
             auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
             this->output_node_dims = tensor_info.GetShape();
         }
        
        use_cuda = false;
        std::cout << "CPU execution provider initialized successfully" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cout << "CPU initialization failed: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cout << "CPU initialization failed: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

void onnxModel::cleanup() {
    session.reset();
}

std::vector<float> onnxModel::predict(std::vector<float>& input_tensor_values, int batch_size, int index)
{
    if (!session) {
        throw std::runtime_error("Session not initialized");
    }
    
    this->input_node_dims[0] = batch_size;
    this->output_node_dims[0] = batch_size;
    float* floatarr = nullptr;
    try {
        std::vector<const char*>output_node_names;
        if (index != -1) {
            output_node_names = { this->output_node_names[index] };
        }
        else {
            output_node_names = this->output_node_names;
        }
        this->input_node_dims[0] = batch_size;
        auto input_tensor_size = input_tensor_values.size();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        floatarr = output_tensors.front().GetTensorMutableData<float>();
    } catch (Ort::Exception& e) {
        throw e;
    }
    int64_t output_tensor_size = 1;
    for (auto& it : this->output_node_dims) {
        output_tensor_size *= it;
    }
    std::vector<float>results(output_tensor_size);
    for (unsigned i = 0; i < output_tensor_size; i++) {
        results[i] = floatarr[i];
    }
    return results;
}

cv::Mat onnxModel::predict(cv::Mat& image, int batch_size, int index)
{
    cv::Mat input_tensor;
    cv::resize(image, input_tensor, {1024, 1024}, 0.0, 0.0, cv::INTER_LANCZOS4);
    cv::cvtColor(input_tensor, input_tensor, cv::COLOR_BGR2RGB);

    int input_tensor_size = input_tensor.cols * input_tensor.rows * 3;
    std::size_t counter = 0;
    std::vector<float>input_data(input_tensor_size);
    std::vector<float>output_data;
    try {
        for (unsigned k = 0; k < 3; k++) {
            for (unsigned i = 0; i < input_tensor.rows; i++) {
                for (unsigned j = 0; j < input_tensor.cols; j++) {
                    input_data[counter++] = static_cast<float>(input_tensor.at<cv::Vec3b>(i, j)[k]) / 255.0;
                }
            }
        }
    } catch (cv::Exception& e) {
        printf(e.what());
    } try {
        output_data = this->predict(input_data);
    } catch (Ort::Exception& e) {
        throw e;
    }

    cv::Mat output_tensor(output_data);
    output_tensor = 255.0 - output_tensor.reshape(1, {1024, 1024}) * 255.0;
    cv::threshold(output_tensor, output_tensor, 220, 255, cv::THRESH_BINARY_INV);
    cv::resize(output_tensor, output_tensor, {image.cols, image.rows}, 0, 0, cv::INTER_LANCZOS4);
    output_tensor.convertTo(output_tensor, CV_8U);

    cv::Mat result;
    image.copyTo(result, output_tensor);
    return result;
}

extern "C" {
    OnnxModel* OnnxModel_new(const char* onnx_model_path) {
        try {
            std::wstring wpath(onnx_model_path, onnx_model_path + strlen(onnx_model_path));
            return new onnxModel(wpath.c_str());
        } catch (const std::exception& e) {
            std::cerr << "Exception in OnnxModel_new: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Unknown exception in OnnxModel_new" << std::endl;
            return nullptr;
        }
    }

    unsigned char* OnnxModel_predict(OnnxModel* model, const char* data, int size, int* out_size) {
        try {
            std::vector<char> buffer(data, data + size);
            cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);
            if (img.empty()) {
                return nullptr;
            }

            cv::Mat output = model->predict(img);

            std::vector<unsigned char> resultBuffer;
            cv::imencode(".png", output, resultBuffer);

            unsigned char* result = (unsigned char*)malloc(resultBuffer.size());
            std::memcpy(result, resultBuffer.data(), resultBuffer.size());

            *out_size = resultBuffer.size();
            return result;
        } catch (const std::exception& e) {
            std::cerr << "Exception in OnnxModel_predict: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Unknown exception in OnnxModel_predict" << std::endl;
            return nullptr;
        }
    }

    void OnnxModel_delete(OnnxModel* model) {
        delete model;
    }

    void free_malloc(void *ptr) {
        free(ptr);
    };
}