#include "lab1.h"
static const unsigned W = 1366;
static const unsigned H = 768;
static const unsigned NFRAME = 240;

#define N_PARTICLES 10000
#define SCALE 1 / 100
#define INIT_DISTANCE 500
// typedef double2 Coord;
// typedef double2 Velocity;
// typedef short3* Velocity;



struct Lab1VideoGenerator::Impl {
	int t = 0;
    double2* coordinate;
    double2* velocity;
    short3* canvas;
};



__global__ void fillKernel(short3* canvas, float alpha) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < W * H) {
        canvas[i] = make_short3(0, 0, 0);
    }
}
    
void fill(short3* canvas, float alpha=1.0){
    fillKernel<<<((W * H + 255)/ 256),256>>>(canvas, alpha); 
}

__global__ void rgb2yuvKernel(short3* canvas, uint8_t* yuv) {
    auto x = blockDim.x * blockIdx.x + threadIdx.x;
    auto y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < W && y < H) {
        auto c = canvas[y * W + x];
        yuv[y * W + x] = 0.299*c.x + 0.587*c.y + 0.114*c.z;
        if (x % 2 == 0 && y % 2 == 0) {
            auto c2 = canvas[(y + 0) * W + x + 1];
            auto c3 = canvas[(y + 1) * W + x + 0];
            auto c4 = canvas[(y + 1) * W + x + 1];
            c.x = (c.x + c2.x + c3.x + c4.x) / 4;
            c.y = (c.y + c2.y + c3.y + c4.y) / 4;
            c.z = (c.z + c2.z + c3.z + c4.z) / 4;
            auto indU = W*H + y/2 * W/2 + x/2;
            auto indV = W*H + W*H/4 + y/2 * W/2 + x/2;
            yuv[indU] = -0.169*c.x - 0.331*c.y + 0.500*c.z + 128; 
            yuv[indV] = 0.500*c.x - 0.419*c.y - 0.081*c.z + 128; 
        }
    }    
}

void rgb2yuv(short3* canvas, uint8_t* yuv) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid((W + 15)/16, (H + 15)/16);
    rgb2yuvKernel<<<dimGrid, dimBlock>>>(canvas, yuv);    
}

__device__ void drawDot(double2 coord, short3 color, short3* canvas) {
    unsigned int x = coord.x * SCALE;
    unsigned int y = coord.y * SCALE;
    canvas[(y + 0)*W + x + 0] = color;
    // canvas[(y + 0)*W + x + 1] = color;
    // canvas[(y + 1)*W + x + 0] = color;
    // canvas[(y + 1)*W + x + 1] = color;
}


__global__ void initParticlesKernel(double2* coord, double2* velocity, short3* canvas) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_PARTICLES) {
        int x = i % 100;
        int y = i / 100;
        coord[i] = make_double2(x * INIT_DISTANCE, y * INIT_DISTANCE);
        velocity[i] = make_double2(0, 0);
    }
}


void initParticles(double2* coord, double2* velocity, short3* canvas) {
    initParticlesKernel<<<(N_PARTICLES + 255) / 256, 256>>>(coord, velocity, canvas);
}


__global__ void updateParticlesKernel(double2* coord, double2* velocity, short3* canvas) {
    const short3 WHITE = make_short3(255, 255, 255);

    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_PARTICLES) {
        // coord[i] = make_double2(x * INIT_DISTANCE, y * INIT_DISTANCE);
        // velocity[i] = make_double2(0, 0);
        drawDot(coord[i], WHITE, canvas);
    }
}


void updateParticles(double2* coord, double2* velocity, short3* canvas) {
    updateParticlesKernel<<<(N_PARTICLES + 31) / 32, 32>>>(coord, velocity, canvas);
}


Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
    cudaMalloc(&(impl->velocity), sizeof(double2) * N_PARTICLES);
    cudaMalloc(&(impl->coordinate), sizeof(double2) * N_PARTICLES);
    cudaMalloc(&(impl->canvas), sizeof(short3) * W * H * 3 / 2);
    fill(impl->canvas);
    initParticles(impl->coordinate, impl->velocity, impl->canvas);
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	// cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	// cudaMemset(yuv+W*H, 128, W*H/2);
    fill(impl->canvas);
    updateParticles(impl->coordinate, impl->velocity, impl->canvas);
    rgb2yuv(impl->canvas, yuv);
	++(impl->t);
}
