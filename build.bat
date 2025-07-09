@echo off
echo Building bilibili-vtuber with CUDA support...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
cmake ..

REM Build the project
cmake --build . --config Release

echo Build completed!
echo The DLL can now run on systems with or without CUDA installed.
pause 