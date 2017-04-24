#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


struct one_if_not_space : public thrust::unary_function<char,int>
{
  __host__ __device__
  int operator()(char c) { return c == '\n' ? 0 : 1; }
};


void CountPosition1(const char *text, int *pos, int text_size)
{
    one_if_not_space func;
    thrust::transform(thrust::device, text, text + text_size, pos, func);    
    thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);
}


__global__ void countPositionKernel(const char *text, int *pos, int text_size) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;

    if (text[i] == '\n') {
        pos[i] = 0;
    }
    
    if (i == 0 || text[i - 1] == '\n') {
        int cnt = 0;
        while (i + cnt < text_size && text[i + cnt] != '\n') {
            pos[i + cnt] = cnt + 1;
            cnt += 1;
        }
    }

}

void CountPosition2(const char *text, int *pos, int text_size)
{
    countPositionKernel<<< (text_size + 255)/ 256, 256>>>(text, pos, text_size);
    // one_if_not_space func;
    // thrust::transform(text, text + text_size, pos, func);
    // thrust::inclusive_scan_by_key(pos, pos + text_size, pos, pos);

}
