__global__ void sgemm_naive(
    const __restrict__ float* A, const __restrict__ float* B, __restrict__ float* C,
    int M, int N, int K, float alpha, float beta) {

        // Get position of thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < K) {
            // Loop over shared dim
            cumalitive_sum = 0.0f
            for (int i = 0; i < N; i++) {
                cumalitive_sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = (alpha * cumalitive_sum) + (beta * C[row * K + col]);
        }
    }