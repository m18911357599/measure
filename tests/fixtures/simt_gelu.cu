__global__ void gelu_kernel(float* out, const float* in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float smem[256];
  if (idx >= n) return;
  smem[threadIdx.x] = in[idx];
  __syncthreads();
  float x = smem[threadIdx.x];
  out[idx] = x * 0.5f * (1.0f + erff(x * 0.7071f));
}

void launch_gelu(float* out, const float* in, int n) {
  cudaMemcpy(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice);
  gelu_kernel<<<1, 256>>>(out, in, n);
  cudaDeviceSynchronize();
}
