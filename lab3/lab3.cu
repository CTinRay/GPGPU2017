#include "lab3.h"
#include <cstdio>

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define N_HIER 2

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
    float *buf1, float *buf2, const int wt, const int ht) {
    
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
            buf2[3 * (xt + yt * wt) + color] = sum / nNeighbor;
        }
    }
}


__global__ void DownSample(const float *buf1, float *buf2,
                           const int w, const int h, const int rate,
                           const int nColors) {
    
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < w / nColors / rate and y < h / nColors  / rate) {
        int xMax = MIN((x + 1) * rate, w);
        int yMax = MIN((y + 1) * rate, h);
        float sum[] = {0, 0, 0};
        for (int i = x * rate; i < xMax; ++i) {
            for (int j = y * rate; j < yMax; ++j) {
                for (int c = 0; c < nColors; ++c) {
                    sum[c] += buf1[nColors * (x + y * w) + c];
                }
            }
        }
        for (int c = 0; c < nColors; ++c) {
            buf2[nColors * (x + y * (w / rate)) + c] =
                sum[c] / (xMax - x) / (yMax - y);
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


void PoissonImagecloningSub(
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
    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    
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

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox) {

    
    float *prevBackground;
    cudaMalloc(&prevBackground, 3*wt*ht*sizeof(float));
    cudaMemcpy(prevBackground, background, sizeof(float)*3*wt*ht,
               cudaMemcpyDeviceToDevice);
    
    for (int i = 0; i < N_HIER; ++i) {
        int rate = 1 << (N_HIER - i);
        
        float *subBackground, *subTarget, *subMask;
        cudaMalloc(&subBackground, 3*(wt/rate)*(ht/rate)*sizeof(float));
        cudaMalloc(&subTarget, 3*(wt/rate)*(ht/rate)*sizeof(float));
        cudaMalloc(&subMask, (wt/rate)*(ht/rate)*sizeof(float));
        
        dim3 gdim, bdim(32,16);
        
        gdim = dim3(CeilDiv(wb,32), CeilDiv(hb,16));
        DownSample<<<gdim, bdim>>>(prevBackground, subBackground, wb, hb, rate, 3);
        
        gdim = dim3(CeilDiv(wt,32), CeilDiv(ht,16)), 
        DownSample<<<gdim, bdim>>>(target, subTarget, wt, ht, rate, 3);
        DownSample<<<gdim, bdim>>>(mask, subMask, wt, ht, rate, 1);
        
        PoissonImagecloningSub(subBackground, subTarget, subMask, output,
                               wb, hb, wt, ht, oy, ox);

        gdim = dim3(CeilDiv(wb,32), CeilDiv(hb,16));
        UpSample<<<gdim, bdim>>>(output, prevBackground, wb, hb, rate, 3);
    }
    
}
