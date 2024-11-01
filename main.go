package main

/*
#cgo LDFLAGS: -L./build/Release -lonnxModel
#include <stdlib.h>
#include <string.h>
#include "src/onnxModel.h"
*/
import "C"
import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"unsafe"
)

func remove(model *C.OnnxModel, imgData []byte) (image.Image, error) {
	var outSize C.int
	result := C.OnnxModel_predict(model, (*C.char)(unsafe.Pointer(&imgData[0])), C.int(len(imgData)), &outSize)
	if result == nil {
		return nil, fmt.Errorf("remove background failed")
	}
	defer C.free_malloc(unsafe.Pointer(result))

	outputData := C.GoBytes(unsafe.Pointer(result), outSize)
	outputImg, err := png.Decode(bytes.NewReader(outputData))
	if err != nil {
		return nil, err
	}
	return outputImg, nil
}

func main() {
	modelPath := C.CString("C:/Users/50728/Documents/code/vtuber-background-remover/.u2net/isnet-o2.onnx")
	defer C.free(unsafe.Pointer(modelPath))
	model := C.OnnxModel_new(modelPath)
	defer C.OnnxModel_delete(model)

	imgFile, err := os.Open("C:/Users/50728/Documents/code/vtuber-background-remover/dataset/o2/imgs/0000.jpg")
	if err != nil {
		fmt.Println("无法打开图片:", err)
		return
	}
	defer imgFile.Close()

	imgData, err := io.ReadAll(imgFile)
	if err != nil {
		fmt.Println("读取图片数据失败:", imgData)
		return
	}

	outputImg, err := remove(model, imgData)
	if err != nil {
		fmt.Println("移除背景失败:", err, outputImg)
		return
	}

	outputFile, err := os.Create("output.png")
	if err != nil {
		fmt.Println("无法创建输出文件:", err)
		return
	}
	defer outputFile.Close()

	err = png.Encode(outputFile, outputImg)
	if err != nil {
		fmt.Println("保存输出图片失败:", err)
		return
	}
}
