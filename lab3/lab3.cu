#include "lab3.h"
#include <cstdio>

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define N_HIER 5
#define N_ITERS 40
#define MAX_W 1

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
        const int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    
        for (int color = 0; color < 3; ++ color){
            float sum = 0;
            for (int d = 0; d < 4; ++d) {
                // calculate target part: - (et + st + wt + nt)
                const int at = xt + directions[d][0];
                const int bt = yt + directions[d][1];
                if (at < wt and at >= 0 and
                    bt < ht and bt >= 0) {
                    sum += target[3 * (wt * yt + xt) + color] -
                         target[3 * (at + wt * bt) + color];

                    // calculate background part 
                    if (at == 0 or at == (wt - 1) or bt == 0 or bt == (ht - 1) or
                        mask[at + wt * bt] <= 127.0f) {
                        const int ab = xt + ox + directions[d][0];
                        const int bb = yt + oy + directions[d][1];
                        sum += background[3 * (ab + bb * wb) + color];
                    }
                }
            }
            fixed[3 * (xt + yt * wt) + color] = sum;
        }
    }
}


__global__ void PoissonImageCloningIteration(
    const float *fixed, const float *mask,
    float *buf1, float *buf2, const int wt, const int ht, float w) {
    
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if (yt < ht and xt < wt) {
        const int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    
        for (int color = 0; color < 3; ++ color){
            // sum = fixed + background part
            float sum = fixed[3 * (xt + yt * wt) + color];
            int nNeighbor = 0;
            for (int d = 0; d < 4; ++d) {
                const int at = xt + directions[d][0];
                const int bt = yt + directions[d][1];
                if (at < wt and at >= 0 and
                    bt < ht and bt >= 0 ) {
                    nNeighbor += 1;
                }
                if (at < wt - 1 and at >= 1 and
                    bt < ht - 1 and bt >= 1 and
                    mask[at + wt * bt] > 128.0f) { 
                    sum += buf1[3 * (at + wt * bt) + color];
                }
            }
            float sor = w * sum / nNeighbor
                + (1 - w)  * buf2[3 * (xt + yt * wt) + color];
            buf2[3 * (xt + yt * wt) + color] = sor;
        }
    }
}


__global__ void DownSample(const float *buf1, float *buf2,
                           const int w, const int h, const int rate,
                           const int nColors) {
    
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < w / rate and y < h / rate) {
        int xMax = MIN((x + 1) * rate, w);
        int yMax = MIN((y + 1) * rate, h);
        float sum[] = {0, 0, 0};
        int n = 0;
        for (int i = x * rate; i < xMax; ++i) {
            for (int j = y * rate; j < yMax; ++j) {
                for (int c = 0; c < nColors; ++c) {
                    sum[c] += buf1[nColors * (i + j * w) + c];
                }
                n += 1;
            }
        }
        for (int c = 0; c < nColors; ++c) {
            // buf2[nColors * (x + y * (w / rate)) + c] =
            //     sum[c] / n;
            buf2[nColors * (x + y * (w / rate)) + c] =
                buf1[nColors * (x * rate + y * rate * w) + c];
        }
    }    
}


__global__ void UpSample(const float *buf1, float *buf2,
                         const int w, const int h, const int rate,
                         const int nColors) {
    
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < w and y < h) {
        for (int c = 0; c < nColors; ++c) {
            buf2[nColors * (x + y * w) + c] =
                buf1[nColors * ((x / rate) + (y / rate) * (w / rate)) + c];
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

    cudaMemset(output, 0, 3*wb*hb*sizeof(float));
    
    float *prevBackground;
    cudaMalloc(&prevBackground, 3*wb*hb*sizeof(float));
    cudaMemcpy(prevBackground, background, sizeof(float)*3*wb*hb,
               cudaMemcpyDeviceToDevice);

    // set up
    float *buf1, *buf2;
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

    
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);

    // use target as initial
    cudaMemcpy(buf2, target, wt*ht*sizeof(float)*3, cudaMemcpyDeviceToDevice);

    for (int i = 0; i < N_HIER; ++i) {
        int rate = 1 << (N_HIER - i - 1);
        // printf("rate = %d\n", rate);
        
        float *fixed, *subBackground, *subTarget, *subMask;
        cudaMalloc(&subBackground, 3*wb*hb*sizeof(float));
        cudaMalloc(&subTarget, 3*(wt/rate)*(ht/rate)*sizeof(float));
        cudaMalloc(&subMask, (wt/rate)*(ht/rate)*sizeof(float));
        cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
        
        // Prepare background to calculate fixed
        gdim = dim3(CeilDiv(wb,32), CeilDiv(hb,16));        
        DownSample<<<gdim, bdim>>>(prevBackground, subBackground, wb, hb, rate, 3);

        // Prepare subMask, subTarget to calculate fixed
        gdim = dim3(CeilDiv(wt,32), CeilDiv(ht,16)), 
        DownSample<<<gdim, bdim>>>(mask, subMask, wt, ht, rate, 1);
        DownSample<<<gdim, bdim>>>(target, subTarget, wt, ht, rate, 3);
        CalculateFixed<<<gdim, bdim>>>(subBackground, subTarget, subMask, fixed,
                                       wb/rate, hb/rate, wt/rate, ht/rate, oy/rate, ox/rate);

        // DownSample from previous result and store in buf1
        DownSample<<<gdim, bdim>>>(buf2, buf1, wt, ht, rate, 3);

        // Do iteration
        for (int i = 0; i < N_ITERS; ++i) {
            float w = ((N_ITERS - i - 1) * MAX_W + i * 1) / (N_ITERS - 1);
            PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, subMask,
                                                         buf1, buf2,
                                                         wt/rate, ht/rate, w);
            PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, subMask,
                                                         buf2, buf1,
                                                         wt/rate, ht/rate, w);
            
        }
        
        // UpSample result and keep it in buf2
        gdim = dim3(CeilDiv(wb,32), CeilDiv(hb,16));
        UpSample<<<gdim, bdim>>>(buf1, buf2, wt, ht, rate, 3);
        
        cudaFree(subBackground);
        cudaFree(subTarget);
        cudaFree(subMask);
        cudaFree(fixed);
    }

    // copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(background, buf2, mask, output,
                                wb, hb, wt, ht, oy, ox);
    
    // clean up
    cudaFree(buf1);
    cudaFree(buf2);    
}
