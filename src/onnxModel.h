#pragma once

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