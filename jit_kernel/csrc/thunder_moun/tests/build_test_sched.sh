nvcc -std=c++17 -O3 \
     -Xcompiler "-std=c++17" \
     -Xcompiler "-D_GLIBCXX_USE_CXX11_ABI=1" \
     -I../../thunder_moun \
     -I../../thunder_moun/tensor \
     test_sched.cc -o test_sched
