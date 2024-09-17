#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#define THREADS_PER_BLOCK 1024
#define SHARED_BLOCK_L 33

__device__ int find(int *parents, int cur) {

    int* start_parent_p = &parents[cur];
    while (parents[cur] != cur) {
        cur = parents[cur];
        *start_parent_p = cur;
    }
    return cur;
}

__device__ void merge(int *parents, int i, int j){

  while(!(i == j)){
    i = find(parents, i);
    j = find(parents, j);

    if (i < j) {
        int cur_j = atomicMin(&parents[j], i);
        if (cur_j == j){
          return;
        }
        else{
          j = cur_j;
        }
    }
    else if (i > j){
        int cur_i = atomicMin(&parents[i], j);
        if (cur_i == i){
          return;
        }
        else{
          i = cur_i;
        }
    }
  }
}

__global__ void initOutput( const float* input,
                            int* output,
                            const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }
  if (input[idx] > 0.5){
    output[idx] = idx;
  }
  else{
    output[idx] = -1;
  }
}

__global__ void mergeOutput(
                            int* output,
                            const int H,
                            const int W,
                            const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }
  if (output[idx] < 0){
    return;
  }

  const int len_per_sample = H * W;
  const int cur_len = idx % len_per_sample;
  const int i = cur_len / W;
  const int j = cur_len % W;

  if (i - 1 >= 0){
    const int idx_neighbor = idx - W;
    if (output[idx_neighbor] >= 0){
      merge(output, idx, idx_neighbor);
    }
  }

  if (j - 1 >= 0){
    const int idx_neighbor = idx - 1;
    if (output[idx_neighbor] >= 0){
      merge(output, idx, idx_neighbor);
    }
  }

  if (i + 1 < H){
    const int idx_neighbor = idx + W;
    if (output[idx_neighbor] >= 0){
      merge(output, idx, idx_neighbor);
    }
  }

  if (j + 1 < W){
    const int idx_neighbor = idx + 1;
    if (output[idx_neighbor] >= 0){
      merge(output, idx, idx_neighbor);
    }
  }
} 

__global__ void  compressOutput(int* output,
                                const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }
  if (output[idx] < 0){
    return;
  }
  output[idx] = find(output, idx);
} 

__global__ void initCount( int* count,
                            const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }
  count[idx] = 0;
}

__global__ void countNum(int* count,
                          int* output,
                          const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }
  if (output[idx] < 0){
    return;
  }
  atomicAdd(&count[output[idx]], 1);
} 

at::Tensor CCForward(const at::Tensor& input,
                                 const int N,
                                 const int H,
                                 const int W) {

  auto output_options = at::TensorOptions().dtype(at::kInt).device(input.device());
  at::Tensor output = at::empty({N, 1, H, W}, output_options);
  at::Tensor count = at::empty({N, 1, H, W}, output_options);

  const int data_len = N * H * W;
  const float *input_vector = input.data<float>();
  int *output_vector = output.data<int>();
  int *count_vector = count.data<int>();

  initOutput<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(input_vector, output_vector, data_len);

  mergeOutput<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(output_vector, H, W, data_len);

  compressOutput<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(output_vector, data_len);

  initCount<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(count_vector, data_len);

  countNum<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(count_vector, output_vector, data_len);
  
  return at::cat({output, count}, 1);
}

__global__ void initDistance( const float* output,
                            int* distance,
                            const int L,
                            const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }
  if (output[idx] > 0.5){
    distance[idx] = 0;
  }
  else{
    distance[idx] = L;
  }
}

__global__ void updateDistance(int* distance,
                            const int L,
                            const int H,
                            const int W,
                            const int data_len) {

  __shared__ int S[SHARED_BLOCK_L][SHARED_BLOCK_L];

  const int block_j = blockIdx.x * L;
  const int block_i = blockIdx.y * L;
  const int j = block_j + threadIdx.x;
  const int i = block_i + threadIdx.y;
  if (i >= H || j >= W){
    return;
  }
  const int n_id = blockIdx.z * H * W;

  S[threadIdx.y][threadIdx.x] = distance[i * W + j + n_id];
  __syncthreads();

  for (int step = 0; step < L - 1; ++step){
    if (threadIdx.y >= 1){
      S[threadIdx.y][threadIdx.x] = min(S[threadIdx.y][threadIdx.x], S[threadIdx.y - 1][threadIdx.x] + 1);
    }
    if (threadIdx.y + 1 < L && i + 1 < H){
      S[threadIdx.y][threadIdx.x] = min(S[threadIdx.y][threadIdx.x], S[threadIdx.y + 1][threadIdx.x] + 1);
    }
    if (threadIdx.x >= 1){
      S[threadIdx.y][threadIdx.x] = min(S[threadIdx.y][threadIdx.x], S[threadIdx.y][threadIdx.x - 1] + 1);
    }
    if (threadIdx.x + 1 < L && j + 1 < W){
      S[threadIdx.y][threadIdx.x] = min(S[threadIdx.y][threadIdx.x], S[threadIdx.y][threadIdx.x + 1] + 1);
    }
    __syncthreads();
  }
  
  distance[i * W + j + n_id] = S[threadIdx.y][threadIdx.x];
}

__global__ void updateDistanceNaive(int* distance,
                            const int H,
                            const int W,
                            const int data_len) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_len){
    return;
  }

  const int len_per_sample = H * W;
  const int cur_len = idx % len_per_sample;
  const int i = cur_len / W;
  const int j = cur_len % W;

  if (i - 1 >= 0){
    const int idx_neighbor = idx - W;
    distance[idx] = min(distance[idx], distance[idx_neighbor] + 1);
  }
  if (i + 1 < H){
    const int idx_neighbor = idx + W;
    distance[idx] = min(distance[idx], distance[idx_neighbor] + 1);
  }
  if (j - 1 >= 0){
    const int idx_neighbor = idx - 1;
    distance[idx] = min(distance[idx], distance[idx_neighbor] + 1);
  }
  if (j + 1 < W){
    const int idx_neighbor = idx + 1;
    distance[idx] = min(distance[idx], distance[idx_neighbor] + 1);
  }
} 

at::Tensor CCBackwardDistance(const at::Tensor& output,
                                 const int L,
                                 const int N,
                                 const int H,
                                 const int W){

  auto distance_options = at::TensorOptions().dtype(at::kInt).device(output.device());
  at::Tensor distance = at::empty({N, 1, H, W}, distance_options);

  const int data_len = N * H * W;
  const float *output_vector = output.data<float>();
  int *distance_vector = distance.data<int>();

  initDistance<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(output_vector, distance_vector, L, data_len);

  int blockDimH = (H + L - 1) / L;
  int blockDimW = (W + L - 1) / L;
  const dim3 gridSize(blockDimW, blockDimH, N);
  const dim3 blockSize(L, L);

  updateDistance<<<gridSize, blockSize, 0, 
    at::cuda::getCurrentCUDAStream()>>>(distance_vector, L, H, W, data_len);
  updateDistanceNaive<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(distance_vector, H, W, data_len);
  updateDistance<<<gridSize, blockSize, 0, 
    at::cuda::getCurrentCUDAStream()>>>(distance_vector, L, H, W, data_len);
  updateDistanceNaive<<<(data_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, 
    at::cuda::getCurrentCUDAStream()>>>(distance_vector, H, W, data_len);
  updateDistance<<<gridSize, blockSize, 0, 
    at::cuda::getCurrentCUDAStream()>>>(distance_vector, L, H, W, data_len);

  return distance;
}
