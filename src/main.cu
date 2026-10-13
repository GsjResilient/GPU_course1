#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(error__));                     \
    }                                                                            \
  } while (0)

constexpr int kImageH = 28;
constexpr int kImageW = 28;
constexpr int kClasses = 10;

struct ModelConfig {
  int kernel{}, c1{}, c2{}, hidden{};
  int conv1_h() const { return kImageH - kernel + 1; }
  int pool1_h() const { return conv1_h() / 2; }
  int conv2_h() const { return pool1_h() - kernel + 1; }
  int pool2_h() const { return conv2_h() / 2; }
  int flat() const { return c2 * pool2_h() * pool2_h(); }
  size_t parameters() const {
    return size_t(c1) * kernel * kernel + c1 +
           size_t(c2) * c1 * kernel * kernel + c2 +
           size_t(hidden) * flat() + hidden + size_t(kClasses) * hidden + kClasses;
  }
};

struct HostWeights {
  ModelConfig config;
  std::vector<float> conv1_w, conv1_b;
  std::vector<float> conv2_w, conv2_b;
  std::vector<float> fc1_w, fc1_b;
  std::vector<float> fc2_w, fc2_b;
};

struct DeviceWeights {
  __half *conv1_w{}, *conv2_w{}, *fc1_w{}, *fc2_w{};
  float *conv1_b{}, *conv2_b{}, *fc1_b{}, *fc2_b{};

  DeviceWeights() = default;
  DeviceWeights(const DeviceWeights &) = delete;
  DeviceWeights &operator=(const DeviceWeights &) = delete;
  DeviceWeights(DeviceWeights &&other) noexcept { *this = std::move(other); }
  DeviceWeights &operator=(DeviceWeights &&other) noexcept {
    if (this != &other) {
      std::swap(conv1_w, other.conv1_w); std::swap(conv1_b, other.conv1_b);
      std::swap(conv2_w, other.conv2_w); std::swap(conv2_b, other.conv2_b);
      std::swap(fc1_w, other.fc1_w); std::swap(fc1_b, other.fc1_b);
      std::swap(fc2_w, other.fc2_w); std::swap(fc2_b, other.fc2_b);
    }
    return *this;
  }

  ~DeviceWeights() {
    cudaFree(conv1_w); cudaFree(conv1_b); cudaFree(conv2_w); cudaFree(conv2_b);
    cudaFree(fc1_w); cudaFree(fc1_b); cudaFree(fc2_w); cudaFree(fc2_b);
  }
};

uint32_t read_be_u32(std::ifstream &in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char *>(b), 4);
  if (!in) throw std::runtime_error("unexpected end of IDX file");
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
         (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

fs::path find_file(const fs::path &root, const std::string &name) {
  const fs::path candidates[] = {
      root / name, root / "raw" / name, root / "MNIST" / "raw" / name};
  for (const auto &p : candidates) if (fs::exists(p)) return p;
  throw std::runtime_error("cannot find " + name + " under " + root.string());
}

void load_mnist(const fs::path &root, std::vector<__half> &images,
                std::vector<uint8_t> &labels, int limit) {
  const auto image_path = find_file(root, "t10k-images-idx3-ubyte");
  const auto label_path = find_file(root, "t10k-labels-idx1-ubyte");
  std::ifstream image_in(image_path, std::ios::binary);
  std::ifstream label_in(label_path, std::ios::binary);
  if (!image_in || !label_in) throw std::runtime_error("failed to open MNIST files");

  const uint32_t image_magic = read_be_u32(image_in);
  const uint32_t image_count = read_be_u32(image_in);
  const uint32_t rows = read_be_u32(image_in);
  const uint32_t cols = read_be_u32(image_in);
  const uint32_t label_magic = read_be_u32(label_in);
  const uint32_t label_count = read_be_u32(label_in);
  if (image_magic != 2051 || label_magic != 2049 || rows != kImageH ||
      cols != kImageW || image_count != label_count) {
    throw std::runtime_error("invalid or incompatible MNIST IDX files");
  }
  const size_t count = limit > 0 ? std::min<size_t>(limit, image_count) : image_count;
  std::vector<uint8_t> bytes(count * kImageH * kImageW);
  labels.resize(count);
  image_in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  label_in.read(reinterpret_cast<char *>(labels.data()), labels.size());
  if (!image_in || !label_in) throw std::runtime_error("truncated MNIST IDX files");

  images.resize(bytes.size());
  constexpr float mean = 0.1307f;
  constexpr float stddev = 0.3081f;
  for (size_t i = 0; i < bytes.size(); ++i)
    images[i] = __float2half((float(bytes[i]) / 255.0f - mean) / stddev);
}

template <typename T>
void read_exact(std::ifstream &in, std::vector<T> &dst, size_t count) {
  dst.resize(count);
  in.read(reinterpret_cast<char *>(dst.data()), count * sizeof(T));
  if (!in) throw std::runtime_error("truncated weights file");
}

HostWeights load_weights(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open weights: " + path.string());
  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "LENETV2", 7) != 0)
    throw std::runtime_error("invalid weights file header");
  HostWeights w;
  uint32_t header[5]{};
  in.read(reinterpret_cast<char *>(header), sizeof(header));
  if (!in || header[0] != 2) throw std::runtime_error("unsupported weights version");
  w.config = {int(header[1]), int(header[2]), int(header[3]), int(header[4])};
  const auto &c = w.config;
  if ((c.kernel != 3 && c.kernel != 5) || c.c1 <= 0 || c.c2 <= 0 ||
      c.hidden <= 0 || c.pool2_h() <= 0)
    throw std::runtime_error("invalid model configuration in weights file");
  read_exact(in, w.conv1_w, c.c1 * 1 * c.kernel * c.kernel);
  read_exact(in, w.conv1_b, c.c1);
  read_exact(in, w.conv2_w, c.c2 * c.c1 * c.kernel * c.kernel);
  read_exact(in, w.conv2_b, c.c2);
  read_exact(in, w.fc1_w, size_t(c.hidden) * c.flat());
  read_exact(in, w.fc1_b, c.hidden);
  read_exact(in, w.fc2_w, kClasses * c.hidden);
  read_exact(in, w.fc2_b, kClasses);

  // Store convolution weights as [K, output_channel]. This layout gives
  // coalesced loads when a thread block stages a GEMM tile in shared memory.
  auto transpose_conv = [](std::vector<float> &weights, int out_c, int k) {
    std::vector<float> transposed(weights.size());
    for (int oc = 0; oc < out_c; ++oc)
      for (int i = 0; i < k; ++i)
        transposed[i * out_c + oc] = weights[oc * k + i];
    weights.swap(transposed);
  };
  transpose_conv(w.conv1_w, c.c1, c.kernel * c.kernel);
  transpose_conv(w.conv2_w, c.c2, c.c1 * c.kernel * c.kernel);
  transpose_conv(w.fc1_w, c.hidden, c.flat());
  transpose_conv(w.fc2_w, kClasses, c.hidden);
  return w;
}

void copy_to_device(float **dst, const std::vector<float> &src) {
  CUDA_CHECK(cudaMalloc(dst, src.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(*dst, src.data(), src.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void copy_to_device_half(__half **dst, const std::vector<float> &src) {
  std::vector<__half> converted(src.size());
  for (size_t i = 0; i < src.size(); ++i) converted[i] = __float2half(src[i]);
  CUDA_CHECK(cudaMalloc(dst, converted.size() * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(*dst, converted.data(), converted.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
}

DeviceWeights upload_weights(const HostWeights &h) {
  DeviceWeights d;
  copy_to_device_half(&d.conv1_w, h.conv1_w); copy_to_device(&d.conv1_b, h.conv1_b);
  copy_to_device_half(&d.conv2_w, h.conv2_w); copy_to_device(&d.conv2_b, h.conv2_b);
  copy_to_device_half(&d.fc1_w, h.fc1_w); copy_to_device(&d.fc1_b, h.fc1_b);
  copy_to_device_half(&d.fc2_w, h.fc2_w); copy_to_device(&d.fc2_b, h.fc2_b);
  return d;
}

// Implicit-im2col Tensor Core convolution. Each warp computes one 16x16 output
// tile with FP16 inputs and FP32 accumulation; im2col is never materialized.
__global__ void conv_relu(const __half *__restrict__ input,
                          const __half *__restrict__ weight,
                          const float *__restrict__ bias,
                          __half *__restrict__ output, int n, int in_c,
                          int in_h, int out_c, int kernel) {
  constexpr int tile = 16;
  constexpr int warps_per_block = 4;
  __shared__ __half a_tiles[warps_per_block][tile * tile];
  __shared__ __half b_tiles[warps_per_block][tile * tile];
  __shared__ float c_tiles[warps_per_block][tile * tile];
  const int out_h = in_h - kernel + 1;
  const int rows = n * out_h * out_h;
  const int kernel_area = kernel * kernel;
  const int inner = in_c * kernel_area;
  const int column_tiles = (out_c + tile - 1) / tile;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int output_tile = blockIdx.x * warps_per_block + warp;
  const int row_tile = output_tile / column_tiles;
  const int col_tile = output_tile % column_tiles;
  __half *a_tile = a_tiles[warp];
  __half *b_tile = b_tiles[warp];
  float *c_tile = c_tiles[warp];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tile, tile, tile,
                         __half, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tile, tile, tile,
                         __half, nvcuda::wmma::row_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tile, tile, tile, float> c_frag;
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  for (int base = 0; base < inner; base += tile) {
    for (int index = lane; index < tile * tile; index += 32) {
      const int local_row = index / tile;
      const int local_k = index % tile;
      const int row = row_tile * tile + local_row;
      const int a_k = base + local_k;
      if (row < rows && a_k < inner) {
        const int batch = row / (out_h * out_h);
        const int spatial = row % (out_h * out_h);
        const int oy = spatial / out_h;
        const int ox = spatial % out_h;
        const int ic = a_k / kernel_area;
        const int kernel_pos = a_k % kernel_area;
        const int iy = oy + kernel_pos / kernel;
        const int ix = ox + kernel_pos % kernel;
        a_tile[index] = input[((batch * in_h + iy) * in_h + ix) * in_c + ic];
      } else {
        a_tile[index] = __float2half(0.0f);
      }
      const int b_k = base + local_row;
      const int oc = col_tile * tile + local_k;
      b_tile[index] = (b_k < inner && oc < out_c)
                          ? weight[b_k * out_c + oc] : __float2half(0.0f);
    }
    __syncwarp();
    nvcuda::wmma::load_matrix_sync(a_frag, a_tile, tile);
    nvcuda::wmma::load_matrix_sync(b_frag, b_tile, tile);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }
  nvcuda::wmma::store_matrix_sync(c_tile, c_frag, tile, nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < tile * tile; index += 32) {
    const int row = row_tile * tile + index / tile;
    const int oc = col_tile * tile + index % tile;
    if (row < rows && oc < out_c)
      output[row * out_c + oc] = __float2half(fmaxf(c_tile[index] + bias[oc], 0.0f));
  }
}

__global__ void maxpool2x2(const __half *__restrict__ input,
                           __half *__restrict__ output, int n, int channels,
                           int in_h) {
  const int out_h = in_h / 2;
  const int total = n * channels * out_h * out_h;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += blockDim.x * gridDim.x) {
    int t = index;
    const int c = t % channels; t /= channels;
    const int ox = t % out_h; t /= out_h;
    const int oy = t % out_h; const int batch = t / out_h;
    const __half *x = input + ((batch * in_h + oy * 2) * in_h + ox * 2) * channels + c;
    const float maximum = fmaxf(fmaxf(__half2float(x[0]), __half2float(x[channels])),
                                fmaxf(__half2float(x[in_h * channels]),
                                      __half2float(x[(in_h + 1) * channels])));
    output[index] = __float2half(maximum);
  }
}

template <bool kRelu, bool kInputNhwc, bool kOutputHalf>
__global__ void linear_wmma(const __half *__restrict__ input,
                            const __half *__restrict__ weight,
                            const float *__restrict__ bias,
                            __half *__restrict__ output_half,
                            float *__restrict__ output_float, int n, int in_features,
                            int out_features, int channels, int spatial_size) {
  constexpr int tile = 16;
  constexpr int warps_per_block = 4;
  __shared__ __half a_tiles[warps_per_block][tile * tile];
  __shared__ __half b_tiles[warps_per_block][tile * tile];
  __shared__ float c_tiles[warps_per_block][tile * tile];
  const int column_tiles = (out_features + tile - 1) / tile;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int output_tile = blockIdx.x * warps_per_block + warp;
  const int row_tile = output_tile / column_tiles;
  const int col_tile = output_tile % column_tiles;
  __half *a_tile = a_tiles[warp];
  __half *b_tile = b_tiles[warp];
  float *c_tile = c_tiles[warp];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tile, tile, tile,
                         __half, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tile, tile, tile,
                         __half, nvcuda::wmma::row_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tile, tile, tile, float> c_frag;
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  for (int base = 0; base < in_features; base += tile) {
    for (int index = lane; index < tile * tile; index += 32) {
      const int local_row = index / tile;
      const int local_k = index % tile;
      const int row = row_tile * tile + local_row;
      const int feature = base + local_k;
      __half value = __float2half(0.0f);
      if (row < n && feature < in_features) {
        if constexpr (kInputNhwc) {
          const int spatial_area = spatial_size * spatial_size;
          const int channel = feature / spatial_area;
          const int spatial = feature % spatial_area;
          value = input[row * in_features + spatial * channels + channel];
        } else {
          value = input[row * in_features + feature];
        }
      }
      a_tile[index] = value;
      const int weight_k = base + local_row;
      const int col = col_tile * tile + local_k;
      b_tile[index] = (weight_k < in_features && col < out_features)
                          ? weight[weight_k * out_features + col] : __float2half(0.0f);
    }
    __syncwarp();
    nvcuda::wmma::load_matrix_sync(a_frag, a_tile, tile);
    nvcuda::wmma::load_matrix_sync(b_frag, b_tile, tile);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }
  nvcuda::wmma::store_matrix_sync(c_tile, c_frag, tile, nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < tile * tile; index += 32) {
    const int row = row_tile * tile + index / tile;
    const int col = col_tile * tile + index % tile;
    if (row < n && col < out_features) {
      const float value = c_tile[index] + bias[col];
      const float activated = kRelu ? fmaxf(value, 0.0f) : value;
      if constexpr (kOutputHalf)
        output_half[row * out_features + col] = __float2half(activated);
      else
        output_float[row * out_features + col] = activated;
    }
  }
}

// Fully fused path for the small k=3 networks selected by the architecture
// search. One block owns one image, keeps every intermediate activation in
// shared memory, and writes only the final ten logits to global memory.
__global__ void fused_small_cnn(const __half *__restrict__ input,
                                const __half *__restrict__ conv1_w,
                                const float *__restrict__ conv1_b,
                                const __half *__restrict__ conv2_w,
                                const float *__restrict__ conv2_b,
                                const __half *__restrict__ fc1_w,
                                const float *__restrict__ fc1_b,
                                const __half *__restrict__ fc2_w,
                                const float *__restrict__ fc2_b,
                                float *__restrict__ logits, int c1, int c2,
                                int hidden) {
  constexpr int conv1_h = 26;
  constexpr int pool1_h = 13;
  constexpr int conv2_h = 11;
  constexpr int pool2_h = 5;
  extern __shared__ __half storage[];
  __half *image = storage;
  __half *conv1 = image + 28 * 28;
  __half *pool1 = conv1 + conv1_h * conv1_h * c1;
  __half *conv2 = pool1 + pool1_h * pool1_h * c1;
  __half *pool2 = conv2 + conv2_h * conv2_h * c2;
  __half *fc1 = pool2 + pool2_h * pool2_h * c2;
  const int tid = threadIdx.x;

  for (int i = tid; i < 28 * 28; i += blockDim.x)
    image[i] = input[blockIdx.x * 28 * 28 + i];
  __syncthreads();

  const int conv1_elements = conv1_h * conv1_h * c1;
  for (int index = tid; index < conv1_elements; index += blockDim.x) {
    const int oc = index % c1;
    const int spatial = index / c1;
    const int oy = spatial / conv1_h;
    const int ox = spatial % conv1_h;
    float sum = conv1_b[oc];
    #pragma unroll
    for (int ky = 0; ky < 3; ++ky)
      #pragma unroll
      for (int kx = 0; kx < 3; ++kx)
        sum = fmaf(__half2float(image[(oy + ky) * 28 + ox + kx]),
                   __half2float(conv1_w[(ky * 3 + kx) * c1 + oc]), sum);
    conv1[index] = __float2half(fmaxf(sum, 0.0f));
  }
  __syncthreads();

  const int pool1_elements = pool1_h * pool1_h * c1;
  for (int index = tid; index < pool1_elements; index += blockDim.x) {
    const int channel = index % c1;
    const int spatial = index / c1;
    const int oy = spatial / pool1_h;
    const int ox = spatial % pool1_h;
    const __half *x = conv1 + ((oy * 2) * conv1_h + ox * 2) * c1 + channel;
    const float maximum = fmaxf(fmaxf(__half2float(x[0]), __half2float(x[c1])),
                                fmaxf(__half2float(x[conv1_h * c1]),
                                      __half2float(x[(conv1_h + 1) * c1])));
    pool1[index] = __float2half(maximum);
  }
  __syncthreads();

  const int conv2_elements = conv2_h * conv2_h * c2;
  for (int index = tid; index < conv2_elements; index += blockDim.x) {
    const int oc = index % c2;
    const int spatial = index / c2;
    const int oy = spatial / conv2_h;
    const int ox = spatial % conv2_h;
    float sum = conv2_b[oc];
    for (int ic = 0; ic < c1; ++ic) {
      #pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          const int activation = ((oy + ky) * pool1_h + ox + kx) * c1 + ic;
          const int weight = (ic * 9 + ky * 3 + kx) * c2 + oc;
          sum = fmaf(__half2float(pool1[activation]),
                     __half2float(conv2_w[weight]), sum);
        }
      }
    }
    conv2[index] = __float2half(fmaxf(sum, 0.0f));
  }
  __syncthreads();

  const int pool2_elements = pool2_h * pool2_h * c2;
  for (int index = tid; index < pool2_elements; index += blockDim.x) {
    const int channel = index % c2;
    const int spatial = index / c2;
    const int oy = spatial / pool2_h;
    const int ox = spatial % pool2_h;
    const __half *x = conv2 + ((oy * 2) * conv2_h + ox * 2) * c2 + channel;
    const float maximum = fmaxf(fmaxf(__half2float(x[0]), __half2float(x[c2])),
                                fmaxf(__half2float(x[conv2_h * c2]),
                                      __half2float(x[(conv2_h + 1) * c2])));
    pool2[index] = __float2half(maximum);
  }
  __syncthreads();

  const int flat = pool2_h * pool2_h * c2;
  for (int neuron = tid; neuron < hidden; neuron += blockDim.x) {
    float sum = fc1_b[neuron];
    for (int feature = 0; feature < flat; ++feature) {
      const int channel = feature / (pool2_h * pool2_h);
      const int spatial = feature % (pool2_h * pool2_h);
      sum = fmaf(__half2float(pool2[spatial * c2 + channel]),
                 __half2float(fc1_w[feature * hidden + neuron]), sum);
    }
    fc1[neuron] = __float2half(fmaxf(sum, 0.0f));
  }
  __syncthreads();

  for (int digit = tid; digit < kClasses; digit += blockDim.x) {
    float sum = fc2_b[digit];
    for (int feature = 0; feature < hidden; ++feature)
      sum = fmaf(__half2float(fc1[feature]),
                 __half2float(fc2_w[feature * kClasses + digit]), sum);
    logits[blockIdx.x * kClasses + digit] = sum;
  }
}

struct Workspace {
  __half *input{}, *conv1{}, *pool1{}, *conv2{}, *pool2{}, *fc1{};
  float *logits{};
  explicit Workspace(int n, const ModelConfig &c) {
    CUDA_CHECK(cudaMalloc(&input, size_t(n) * 28 * 28 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&conv1, size_t(n) * c.c1 * c.conv1_h() * c.conv1_h() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&pool1, size_t(n) * c.c1 * c.pool1_h() * c.pool1_h() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&conv2, size_t(n) * c.c2 * c.conv2_h() * c.conv2_h() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&pool2, size_t(n) * c.c2 * c.pool2_h() * c.pool2_h() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&fc1, size_t(n) * c.hidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&logits, size_t(n) * kClasses * sizeof(float)));
  }
  ~Workspace() {
    cudaFree(input); cudaFree(conv1); cudaFree(pool1); cudaFree(conv2);
    cudaFree(pool2); cudaFree(fc1); cudaFree(logits);
  }
};

int blocks_for(size_t work) {
  constexpr int threads = 256;
  return std::min<size_t>((work + threads - 1) / threads, 65535);
}

void forward(const Workspace &x, const DeviceWeights &w, const ModelConfig &c, int n) {
  if (c.kernel == 3 && c.c1 <= 8 && c.c2 <= 8 && c.hidden <= 16) {
    const size_t shared_elements = 28 * 28 + 26 * 26 * c.c1 +
        13 * 13 * c.c1 + 11 * 11 * c.c2 + 5 * 5 * c.c2 + c.hidden;
    fused_small_cnn<<<n, 256, shared_elements * sizeof(__half)>>>(
        x.input, w.conv1_w, w.conv1_b, w.conv2_w, w.conv2_b,
        w.fc1_w, w.fc1_b, w.fc2_w, w.fc2_b, x.logits,
        c.c1, c.c2, c.hidden);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  constexpr int threads = 256;
  constexpr int tile = 16;
  constexpr int warps_per_block = 4;
  const int conv1_tiles = ((n * c.conv1_h() * c.conv1_h() + tile - 1) / tile) *
                          ((c.c1 + tile - 1) / tile);
  const int conv2_tiles = ((n * c.conv2_h() * c.conv2_h() + tile - 1) / tile) *
                          ((c.c2 + tile - 1) / tile);
  conv_relu<<<(conv1_tiles + warps_per_block - 1) / warps_per_block, 128>>>(
      x.input, w.conv1_w, w.conv1_b, x.conv1, n, 1, 28, c.c1, c.kernel);
  maxpool2x2<<<blocks_for(size_t(n) * c.c1 * c.pool1_h() * c.pool1_h()), threads>>>(
      x.conv1, x.pool1, n, c.c1, c.conv1_h());
  conv_relu<<<(conv2_tiles + warps_per_block - 1) / warps_per_block, 128>>>(
      x.pool1, w.conv2_w, w.conv2_b, x.conv2, n, c.c1, c.pool1_h(), c.c2, c.kernel);
  maxpool2x2<<<blocks_for(size_t(n) * c.c2 * c.pool2_h() * c.pool2_h()), threads>>>(
      x.conv2, x.pool2, n, c.c2, c.conv2_h());
  const int fc1_tiles = ((n + tile - 1) / tile) * ((c.hidden + tile - 1) / tile);
  const int fc2_tiles = ((n + tile - 1) / tile) * ((kClasses + tile - 1) / tile);
  linear_wmma<true, true, true><<<(fc1_tiles + warps_per_block - 1) / warps_per_block, 128>>>(
      x.pool2, w.fc1_w, w.fc1_b, x.fc1, nullptr, n, c.flat(), c.hidden, c.c2, c.pool2_h());
  linear_wmma<false, false, false><<<(fc2_tiles + warps_per_block - 1) / warps_per_block, 128>>>(
      x.fc1, w.fc2_w, w.fc2_b, nullptr, x.logits, n, c.hidden, kClasses, 0, 0);
  CUDA_CHECK(cudaGetLastError());
}

struct Options {
  fs::path data = "data";
  fs::path weights = "weights/lenet.bin";
  int batch_size = 10000;
  int warmup = 3;
  int repeats = 1;
  int limit = 0;
};

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + a);
      return argv[i];
    };
    if (a == "--data") o.data = value();
    else if (a == "--weights") o.weights = value();
    else if (a == "--batch-size") o.batch_size = std::stoi(value());
    else if (a == "--warmup") o.warmup = std::stoi(value());
    else if (a == "--repeats") o.repeats = std::stoi(value());
    else if (a == "--limit") o.limit = std::stoi(value());
    else if (a == "--help" || a == "-h") {
      std::cout << "Usage: mnist_cuda [--data DIR] [--weights FILE] "
                   "[--batch-size N] [--warmup N] [--repeats N] [--limit N]\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.batch_size <= 0 || o.warmup < 0 || o.repeats <= 0 || o.limit < 0)
    throw std::runtime_error("batch/repeats must be positive; warmup/limit non-negative");
  return o;
}

int main(int argc, char **argv) {
  try {
    const Options opt = parse_args(argc, argv);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " (compute " << prop.major << '.'
              << prop.minor << ")\n";

    std::vector<__half> images;
    std::vector<uint8_t> labels;
    load_mnist(opt.data, images, labels, opt.limit);
    HostWeights host_weights = load_weights(opt.weights);
    const ModelConfig config = host_weights.config;
    std::cout << "Model: k=" << config.kernel << ", c1=" << config.c1
              << ", c2=" << config.c2 << ", hidden=" << config.hidden
              << ", parameters=" << config.parameters() << "\n";
    DeviceWeights weights = upload_weights(host_weights);
    Workspace workspace(opt.batch_size, config);
    std::vector<float> logits(size_t(opt.batch_size) * kClasses);

    const int warm_n = std::min<int>(opt.batch_size, labels.size());
    CUDA_CHECK(cudaMemcpy(workspace.input, images.data(),
                          size_t(warm_n) * 28 * 28 * sizeof(__half),
                          cudaMemcpyHostToDevice));
    for (int i = 0; i < opt.warmup; ++i) forward(workspace, weights, config, warm_n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    double kernel_ms = 0.0;
    double repeated_kernel_ms = 0.0;
    size_t correct = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    for (size_t begin = 0; begin < labels.size(); begin += opt.batch_size) {
      const int n = std::min<size_t>(opt.batch_size, labels.size() - begin);
      CUDA_CHECK(cudaMemcpy(workspace.input, images.data() + begin * 28 * 28,
                            size_t(n) * 28 * 28 * sizeof(__half),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventRecord(start));
      for (int repeat = 0; repeat < opt.repeats; ++repeat)
        forward(workspace, weights, config, n);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float batch_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&batch_ms, start, stop));
      repeated_kernel_ms += batch_ms;
      kernel_ms += batch_ms / opt.repeats;
      CUDA_CHECK(cudaMemcpy(logits.data(), workspace.logits,
                            size_t(n) * kClasses * sizeof(float),
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        const float *row = logits.data() + i * kClasses;
        const int pred = int(std::max_element(row, row + kClasses) - row);
        correct += pred == labels[begin + i];
      }
    }
    const auto wall_stop = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    const double normalized_wall_ms = wall_ms - repeated_kernel_ms + kernel_ms;
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    const double count = labels.size();
    std::cout << std::fixed << std::setprecision(3)
              << "Samples: " << labels.size() << ", batch size: " << opt.batch_size
              << ", timing repeats: " << opt.repeats << '\n'
              << "Accuracy: " << (100.0 * correct / count) << "% (" << correct << '/'
              << labels.size() << ")\n"
              << "CUDA kernels: " << kernel_ms << " ms total, "
              << (count * 1000.0 / kernel_ms) << " images/s, "
              << (kernel_ms * 1000.0 / count) << " us/image\n"
              << "End-to-end: " << normalized_wall_ms << " ms total, "
              << (count * 1000.0 / normalized_wall_ms) << " images/s\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
