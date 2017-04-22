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

void CountPosition2(const char *text, int *pos, int text_size)
{

    // one_if_not_space func;
    // thrust::transform(text, text + text_size, pos, func);
    // thrust::inclusive_scan_by_key(pos, pos + text_size, pos, pos);

}
