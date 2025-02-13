#include "lab1.h"
static const unsigned W = 1366;
static const unsigned H = 1024;
static const unsigned NFRAME = 480;

#define N_PARTICLES 10000
#define SCALE (1.0 / 20)
#define INIT_DISTANCE 80
#define GRAVITY 10.0
#define G_REPULSION 1.5
#define G_COHESION 1.0
#define LOSS 1.05

// typedef float2 Coord;
// typedef float2 Velocity;
// typedef short3* Velocity;

inline __host__ __device__ void operator*=(float2&a, float b) {
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ void operator/=(float2&a, float b) {
    a.x /= b;
    a.y /= b;
}

inline __host__ __device__ void operator+=(float2&a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}


inline __host__ __device__  float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__  float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__  float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__  float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}

inline __host__ __device__  short3 operator*(float a, short3 b) {
    return make_short3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__  short3 operator+(short3 a, short3 b) {
    return make_short3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__device__ float norm(float2 v) {
    return sqrt(v.x * v.x + v.y * v.y);
}


struct Lab1VideoGenerator::Impl {
	int t = 0;
    float2* coordinate;
    float2* velocity;
    float2* prev_coordinate;
    float2* prev_velocity;
    short3* canvas;
};



__global__ void fillKernel(short3* canvas, float alpha) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < W * H) {        
        canvas[i] = (1 - alpha) * canvas[i] + alpha * make_short3(0, 0, 0);
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

__device__ void drawDot(float2 coord, short3 color, short3* canvas) {
    int x = coord.x * SCALE + 20;
    int y = coord.y * SCALE;
    if ( x >= 0 && x < W - 1 && y >= 0 && y < H - 1) {
        canvas[(y + 0)*W + x + 0] = color;
        canvas[(y + 0)*W + x + 1] = color;
        canvas[(y + 1)*W + x + 0] = color;
        canvas[(y + 1)*W + x + 1] = color;
    }
}


__global__ void initParticlesKernel(float2* coord, float2* velocity, short3* canvas) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    int xStart = (W/SCALE - INIT_DISTANCE*100) / 2;
    if (i < N_PARTICLES) {
        int x = i % 100;
        int y = i / 100;
        coord[i] = make_float2(xStart + x * INIT_DISTANCE, y * INIT_DISTANCE);
        velocity[i] = make_float2(0, 0);
    }
}


void initParticles(float2* coord, float2* velocity, short3* canvas) {
    initParticlesKernel<<<(N_PARTICLES + 255) / 256, 256>>>(coord, velocity, canvas);
}


__device__ short3 acce2color(float2 acce) {
    // float acceNorm = acce.x * acce.x + acce.y * acce.y;
    float acceNorm = norm(acce);
    short3 color;
    float threshold = GRAVITY;
    float sig = (1 / (1 + expf(threshold - acceNorm)) - 0.5) * 255 * 4;
    if (sig > 0) {
        if (sig > 255) {
            color.x = 255;
            color.y = 255 - (sig - 255);
        } else {
            color.x = sig;
            color.y = 255;
        }
        color.z = 0;
    } else {
        if (sig > -255) {
            color.z = -sig;
            color.y = 255;
        } else {
            color.y = 255 + (sig + 255);
            color.z = 255;

        }        
        color.x = 0;
    }
    return color;
}


__global__ void updateParticlesKernel(float2* prev_coord, float2* prev_velocity,
                                      float2* coord, float2* velocity, short3* canvas) {
    const short3 WHITE = make_short3(255, 255, 255);

    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_PARTICLES) {
        velocity[i] = prev_velocity[i];
        coord[i] = prev_coord[i];
        float2 acce = make_float2(0, 0);

        // reflection
        if (coord[i].y >= (H - 15)/SCALE) {
            acce.y -= 2 * prev_velocity[i].y;
        }
        if (coord[i].x >= (W - 15)/SCALE) {
            acce.x -= 2 * prev_velocity[i].x;
        }
        if (coord[i].x <= 15/SCALE) {
            acce.x -= 2 * prev_velocity[i].x;
        }

        // calculate acceleration
        acce.y += GRAVITY;
        for (int j = 0; j < N_PARTICLES; ++j) {
            float2 d = prev_coord[i] - prev_coord[j];
            double d2 = (double)(d.x * d.x + d.y * d.y);
            if (d2 < 1e-5) {
                d2 = 1e-5;
            }
            float2 a = d / sqrt(d2) * (G_REPULSION / d2 - G_COHESION / sqrt(d2));
            acce += a;
        }
        acce.x -= G_COHESION / (prev_coord[i].x - 15);
        acce.x += G_COHESION / (W - 15 - prev_coord[i].x);
        acce.y += G_COHESION / (prev_coord[i].y - 15);

        
        // update velocity
        velocity[i] += acce / LOSS;
        
        // update coordinate
        coord[i] += velocity[i] / 8;
        
        // boundary
        coord[i].x = coord[i].x < 10/SCALE ? 10/SCALE : coord[i].x;
        coord[i].x = coord[i].x > (W - 10)/SCALE ? (W - 10)/SCALE : coord[i].x;
        coord[i].y = coord[i].y > (H - 10)/SCALE ? (H - 10)/SCALE : coord[i].y;
        
        short3 color = acce2color(acce);
        drawDot(coord[i], color, canvas);
    }
}


void updateParticles(float2* prev_coord, float2* prev_velocity,
                     float2* coord, float2* velocity, short3* canvas) {
    updateParticlesKernel<<<(N_PARTICLES + 63) / 64, 64>>>(prev_coord, prev_velocity,
                                                           coord, velocity, canvas);
}


Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
    cudaMalloc(&(impl->velocity), sizeof(float2) * N_PARTICLES);
    cudaMalloc(&(impl->coordinate), sizeof(float2) * N_PARTICLES);
    cudaMalloc(&(impl->prev_velocity), sizeof(float2) * N_PARTICLES);
    cudaMalloc(&(impl->prev_coordinate), sizeof(float2) * N_PARTICLES);
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
    for (int i = 0; i < 8; ++i) {
        fill(impl->canvas, 0.15);
        cudaMemcpy(impl->prev_coordinate, impl->coordinate,
                   sizeof(float2) * N_PARTICLES, cudaMemcpyDeviceToDevice);
        cudaMemcpy(impl->prev_velocity, impl->velocity,
                   sizeof(float2) * N_PARTICLES, cudaMemcpyDeviceToDevice);
        updateParticles(impl->prev_coordinate, impl->prev_velocity,
                        impl->coordinate, impl->velocity, impl->canvas);    
        rgb2yuv(impl->canvas, yuv);
    }
    ++(impl->t);
}
