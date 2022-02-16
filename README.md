## Prerequisites

- CMake version 3.17 at least,
- conan package manager,
- arrrus package version 0.6.6 (make sure arrus/lib64 is set in the Path environment variable)

## How to build the application

Set valid paths in CMakeLists.txt: https://github.com/us4useu/pwi_cpp_example/blob/master/CMakeLists.txt#L52-L54

```
git clone https://github.com/us4useu/pwi_cpp_example.git
mkdir build
cd build
conan install ..
cmake ..
cmake --build . --config RelWithDebInfo
```
Then run the application:

```
cd RelWithDebInfo
./pwi_example.exe
```

Top stop the application: To stop the application, press `q` while the window with B-mode is displayed.
