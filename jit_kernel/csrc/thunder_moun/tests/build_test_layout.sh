nvcc -std=c++17 -O3 \
     -I../../thunder_moun \
     -I../../thunder_moun/tensor \
     test_layout.cc -o test_layout

# ./test_layout
