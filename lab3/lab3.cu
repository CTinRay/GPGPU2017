#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}


__global__ void CalculateFixed(const float *background,
                               const float *target,
                               const float *mask,
                               float *fixed,
                               const int wb, const int hb,
                               const int wt, const int ht,
                               const int oy, const int ox) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;

	if (yt < ht and xt < wt) {
        const int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, 1}};
    
        for (int color = 0; color < 3; ++ color){
            // sum = 4 * ct
            float sum = 4 * target[3 * (wt * yt + xt) + color];
            for (int d = 0; d < 4; ++d) {
                // calculate target part: - (et + st + wt + nt)
                const int at = xt + directions[d][0];
                const int bt = yt + directions[d][1];
                if (at < wt and at >= 0 and
                    bt < ht and bt >= 0) {
                    sum -= target[3 * (at + wt * bt) + color];

                    // calculate background part 
                    if (mask[at + wt * bt] < 128.0f) {
                        const int ab = xt + ox + directions[d][0];
                        const int bb = yt + oy + directions[d][1];
                        int indb = bb + wt * ab;
                        sum += background[indb];
                    }
                }
            }
            fixed[3 * (xt + yt * wt) + color] = sum;
        }
    }
}


__global__ void PoissonImageCloningIteration(
    const float *fixed, const float *mask,
    float *buf1, float *buf2, const int wt, const int ht) {
    
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if (yt < ht and xt < wt) {
        const int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, 1}};
    
        for (int color = 0; color < 3; ++ color){
            // sum = fixed + background part
            float sum = fixed[3 * (xt + yt * wt) + color];
            for (int d = 0; d < 4; ++d) {
                const int at = xt + directions[d][0];
                const int bt = yt + directions[d][1];
                if (at < wt and at >= 0 and
                    bt < ht and bt >= 0 and
                    mask[at + wt * bt] > 128.0f) { 
                    sum += buf1[3 * (at + wt * bt) + color];
                }
            }
            buf2[3 * (xt + yt * wt) + color] = 0.25 * sum;
        }
    }
}


void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox) {
    // set up
    float *fixed, *buf1, *buf2;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    CalculateFixed<<<gdim, bdim>>>(background, target, mask, fixed,
                                   wb, hb, wt, ht, oy, ox);

    // iterate
    for (int i = 0; i < 10000; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf1, buf2, wt, ht);
        PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf2, buf1, wt, ht);
    }

    // copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(background, buf1, mask, output,
                                wb, hb, wt, ht, oy, ox);

    // clean up
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
}
