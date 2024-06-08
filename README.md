## How to build and setup

1. Download opencv and dlib libs. And add them to PATH.

2. Configure and build
```bash
mkdir build
```
```bash
cd build
```
```bash
cmake -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
```
```bash
ninja
```
3. Run
```bash
./CV-SAS
```
