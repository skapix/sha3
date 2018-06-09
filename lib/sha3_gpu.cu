#include "sha3_gpu.h"
#include "helper_cuda.h"
#include "common.h"
#include <cstdlib>
#include <cassert>
#include <limits>

namespace
{

constexpr size_t npos = std::numeric_limits<size_t>::max();

// Array of indicies and rotation values for P and Pi phases.
__constant__ uint8_t g_ppi_aux[25][2];

// Array of indices for ksi phase.
__constant__ uint8_t g_ksi_aux[25][2];

__constant__ uint64_t g_iota_aux[24];

bool inited = false;

void init_constants()
{
  const std::array<std::pair<uint8_t, uint8_t>, 25> h_ppi_aux = {
      {{0, 0},   {6, 44},  {12, 43}, {18, 21}, {24, 14}, {3, 28},  {9, 20}, {10, 3}, {16, 45},
       {22, 61}, {1, 1},   {7, 6},   {13, 25}, {19, 8},  {20, 18}, {4, 27}, {5, 36}, {11, 10},
       {17, 15}, {23, 56}, {2, 62},  {8, 55},  {14, 39}, {15, 41}, {21, 2}}};
  checkCudaErrors(cudaMemcpyToSymbol(g_ppi_aux, h_ppi_aux.data(), h_ppi_aux.size() * sizeof(uint8_t) * 2));

  const std::array<std::pair<uint8_t, uint8_t>, 25> h_ksi_aux = {
      {{1, 2},   {2, 3},   {3, 4},   {4, 0},   {0, 1},   {6, 7},   {7, 8},   {8, 9},   {9, 5},
       {5, 6},   {11, 12}, {12, 13}, {13, 14}, {14, 10}, {10, 11}, {16, 17}, {17, 18}, {18, 19},
       {19, 15}, {15, 16}, {21, 22}, {22, 23}, {23, 24}, {24, 20}, {20, 21}}};
  checkCudaErrors(cudaMemcpyToSymbol(g_ksi_aux, h_ksi_aux.data(), h_ksi_aux.size() * sizeof(uint8_t) * 2));

  const std::array<uint64_t, 24> h_iota_aux = {
      0x0000000000000001L, 0x0000000000008082L, 0x800000000000808aL, 0x8000000080008000L, 0x000000000000808bL,
      0x0000000080000001L, 0x8000000080008081L, 0x8000000000008009L, 0x000000000000008aL, 0x0000000000000088L,
      0x0000000080008009L, 0x000000008000000aL, 0x000000008000808bL, 0x800000000000008bL, 0x8000000000008089L,
      0x8000000000008003L, 0x8000000000008002L, 0x8000000000000080L, 0x000000000000800aL, 0x800000008000000aL,
      0x8000000080008081L, 0x8000000000008080L, 0x0000000080000001L, 0x8000000080008008L};

  checkCudaErrors(cudaMemcpyToSymbol(g_iota_aux, h_iota_aux.data(), h_iota_aux.size() * sizeof(uint64_t)));
}


__device__ uint64_t rotate(uint64_t val, unsigned n) { return val << n | val >> (64 - n); }

__device__ void processState(uint64_t *A)
{
  const size_t t = threadIdx.x;
  const size_t s = threadIdx.x % 5;

  __shared__ uint64_t C[25];

  assert(t < 25);

#pragma unroll
  for (int round_idx = 0; round_idx < 24; ++round_idx)
  {
    // Thetta phase.
    C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
    A[t] ^= C[s + 5 - 1] ^ rotate(C[s + 1], 1);

    // P and Pi combined phases.
    C[t] = rotate(A[g_ppi_aux[t][0]], g_ppi_aux[t][1]);

    // Ksi phase.
    A[t] = C[t] ^ (~C[g_ksi_aux[t][0]] & C[g_ksi_aux[t][1]]);

    // Iota phase.
    A[t] ^= t == 0 ? g_iota_aux[round_idx] : 0;
  }
}

__global__ void processBlockDevice(const uint64_t *data, size_t singleBufSize, const uint64_t *end,
                                   uint64_t *A_original)
{
  const size_t t = threadIdx.x;

  __shared__ uint64_t A[25];

  if (t < 25)
  {
    A[t] = A_original[t];
    for (; data != end; data += singleBufSize)
    {
      if (t < singleBufSize)
      {
        // Apply data to inner state. Nvidia keeps all data in little-endian.
        A[t] ^= data[t];
      }
      processState(A);
    }
    A_original[t] = A[t];
  }
}

void addPadding(uint8_t *d_begin, uint8_t *d_end)
{
  const int maxBuf = 144;
  assert(d_end > d_begin);
  size_t size = d_end - d_begin;
  assert(size <= maxBuf);

  uint8_t buf[maxBuf] = {};
  if (size == 1)
  {
    buf[0] = 0x86;
  }
  else
  {
    buf[0] = 0x06;
    buf[size - 1] = 0x80;
  }

  checkCudaErrors(cudaMemcpy(d_begin, buf, size, cudaMemcpyHostToDevice));
}

} // namespace


SHA3_gpu::~SHA3_gpu()
{
  checkCudaErrors(cudaFree(m_d_blockBuffers));
  checkCudaErrors(cudaFree(m_d_A));
}

SHA3_gpu::SHA3_gpu(size_t size)
  : m_digestSize(size / 8)
{
  assert(m_digestSize * 8 == size);
  if (!inited)
  {
    init_constants();
  }
  checkCudaErrors(cudaMalloc(&m_d_A, 25 * 8));
  m_singleBufSz = 200 - 2 * m_digestSize;

  checkCudaErrors(cudaMalloc(&m_d_blockBuffers, m_singleBufSz * m_nBuffers));
  checkCudaErrors(cudaMemset(m_d_blockBuffers, 0, m_singleBufSz * m_nBuffers));

  init();
}

void SHA3_gpu::init()
{
  checkCudaErrors(cudaMemset(m_d_A, 0, 25 * sizeof(uint64_t)));
  m_bufferOffset = 0;

  m_finished = false;
}

void SHA3_gpu::add(const uint8_t *data, size_t sz)
{
  assert(!m_finished && "Init should be called");
  size_t blockSz = m_nBuffers * m_singleBufSz;
  while (sz != 0)
  {
    if (sz < blockSz - m_bufferOffset)
    {
      checkCudaErrors(cudaMemcpy(m_d_blockBuffers + m_bufferOffset, data, sz, cudaMemcpyHostToDevice));
      m_bufferOffset += sz;
      return;
    }

    size_t dataSize = blockSz - m_bufferOffset;
    checkCudaErrors(cudaMemcpy(m_d_blockBuffers + m_bufferOffset, data, dataSize, cudaMemcpyHostToDevice));
    processBlock(blockSz);
    m_bufferOffset = 0;
    sz -= dataSize;
    data += dataSize;
  }
}

std::vector<uint8_t> SHA3_gpu::digest()
{
  if (!m_finished)
  {
    size_t size = (m_bufferOffset / m_singleBufSz + 1) * m_singleBufSz;
    addPadding(m_d_blockBuffers + m_bufferOffset, m_d_blockBuffers + size);
    processBlock(size);
    m_finished = true;
  }

  std::vector<uint8_t> result(m_digestSize);
  checkCudaErrors(cudaMemcpy(result.data(), m_d_A, m_digestSize, cudaMemcpyDeviceToHost));
  return result;
}

void SHA3_gpu::processBlock(size_t bufSize)
{
  assert(bufSize % m_singleBufSz == 0);
  auto ptr64 = reinterpret_cast<const uint64_t *>(m_d_blockBuffers);
  assert(m_singleBufSz % 8 == 0);
  processBlockDevice<<<1, 32>>>(ptr64, m_singleBufSz / 8, ptr64 + bufSize / 8, m_d_A);
}

//
// SHA3_gpu_batch
//

struct SHA3_gpu_batch::State
{
  uint64_t *d_A;
  uint8_t *d_blockBuffer;
  size_t bufferSize = 0; // Buffer's payload size.
};

namespace
{

__global__ void processBatchBlockDevice(SHA3_gpu_batch::State *data, size_t blockSize)
{
  int t = threadIdx.x;
  int b = blockIdx.x;
  __shared__ uint64_t A[25];

  if (t < 25)
  {
    size_t bufSize = data[b].bufferSize / 8;
    A[t] = data[b].d_A[t];
    const uint64_t *buf = reinterpret_cast<const uint64_t *>(data[b].d_blockBuffer);

    for (; bufSize != 0; bufSize -= blockSize, buf += blockSize)
    {
      if (t < blockSize)
      {
        A[t] ^= buf[t];
      }
      processState(A);
    }
    data[b].d_A[t] = A[t];
  }
}

} // namespace

SHA3_gpu_batch::SHA3_gpu_batch(size_t block)
  : m_digestSize(block / 8)
  , m_singleBlockSize(200 - 2 * m_digestSize)
{
  assert(m_digestSize * 8 == block);
  if (!inited)
  {
    init_constants();
  }

  int device;
  checkCudaErrors(cudaGetDevice(&device));

  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props, device));
  int cores = props.major == 9999 && props.minor == 9999 ? 1 : _ConvertSMVer2Cores(props.major, props.minor);
  cores *= props.multiProcessorCount;
  m_nBlocks = cores / props.warpSize;
  m_states = std::make_unique<State[]>(m_nBlocks);
  checkCudaErrors(cudaMalloc(&m_d_states, m_nBlocks * sizeof(State)));
  size_t aSize = 25 * sizeof(uint64_t);
  size_t available;
  checkCudaErrors(cudaMemGetInfo(&available, nullptr));

  size_t nSingleBuffers = (available - aSize * m_nBlocks) / m_nBlocks / m_singleBlockSize;
  if (nSingleBuffers == 0)
  {
    throw std::logic_error("Not enough memory on gpu device. Please, select another one");
  }

  // std::min takes reference and m_maxBuffers is not placed.
  // Create new value, that equals to m_maxBuffers.
  nSingleBuffers = std::min(nSingleBuffers, size_t(m_maxBuffers));

  m_bufferSize = nSingleBuffers * m_singleBlockSize;

  for (size_t i = 0; i < m_nBlocks; ++i)
  {
    checkCudaErrors(cudaMalloc(&m_states[i].d_A, aSize));
    checkCudaErrors(cudaMalloc(&m_states[i].d_blockBuffer, m_bufferSize));
    assert(m_states[i].d_blockBuffer != nullptr);
  }
}

SHA3_gpu_batch::~SHA3_gpu_batch()
{
  for (size_t i = 0; i < m_nBlocks; ++i)
  {
    checkCudaErrors(cudaFree(m_states[i].d_A));
    checkCudaErrors(cudaFree(m_states[i].d_blockBuffer));
  }
  checkCudaErrors(cudaFree(m_d_states));
}

std::vector<SHA3_gpu_batch::Digest>
    SHA3_gpu_batch::calculate(const std::vector<std::pair<const uint8_t *, size_t>> &datas)
{
  struct LocalState
  {
    size_t idx = npos; // index of processed element
    size_t globalOffset = 0;
  };

  std::vector<SHA3_gpu_batch::Digest> result = prepareResult(datas.size());

  size_t loopSize = std::min<size_t>(m_nBlocks, datas.size());

  std::vector<LocalState> localState(loopSize);

  size_t next = 0;
  size_t finished = 0;
  while (finished < datas.size())
  {
    for (size_t i = 0; i < loopSize; ++i)
    {
      // Task distributing.
      if (localState[i].idx == npos)
      {
        if (next >= datas.size())
        {
          // Nothing to give.
          continue;
        }
        localState[i].idx = next++;
        localState[i].globalOffset = 0;
        checkCudaErrors(cudaMemset(m_states[i].d_A, 0, 25 * 8));
      }

      // Fill buffers.
      auto &from = datas[localState[i].idx];
      size_t nCopy = std::min(from.second - localState[i].globalOffset, m_bufferSize);

      checkCudaErrors(cudaMemcpy(m_states[i].d_blockBuffer, from.first + localState[i].globalOffset, nCopy,
                                 cudaMemcpyHostToDevice));

      if (nCopy != m_bufferSize)
      {
        // We didn't fill the whole buffer => it's data end. We need to add padding to the last block.
        size_t newOffset = (1 + nCopy / m_singleBlockSize) * m_singleBlockSize;
        assert(newOffset <= m_bufferSize);
        addPadding(m_states[i].d_blockBuffer + nCopy, m_states[i].d_blockBuffer + newOffset);

        nCopy = newOffset;
      }

      m_states[i].bufferSize = nCopy;
      localState[i].globalOffset += nCopy;
    }

    launchKernel();

    for (size_t i = 0; i < loopSize; ++i)
    {
      if (localState[i].idx != npos && localState[i].globalOffset > datas[localState[i].idx].second)
      {
        // Collect results
        checkCudaErrors(
            cudaMemcpy(result[localState[i].idx].data(), m_states[i].d_A, m_digestSize, cudaMemcpyDeviceToHost));
        localState[i].idx = npos;
        // Mark state as empty for cases when there is no work to do.
        // This 0 is required for gpu not to perform inappropriate work.
        m_states[i].bufferSize = 0;
        ++finished;
      }
    }
  }
  return result;
}

std::vector<SHA3_gpu_batch::Digest> SHA3_gpu_batch::prepareResult(size_t size)
{
  std::vector<Digest> result;
  result.resize(size);
  for (size_t i = 0; i < size; ++i)
  {
    result[i].resize(m_digestSize);
  }
  return result;
}

void SHA3_gpu_batch::launchKernel()
{
  if (!isLittleEndian())
  {
    for (size_t i = 0; i < m_nBlocks; ++i)
    {
      m_states[i].bufferSize = toLittleEndian(m_states[i].bufferSize);
    }
  }

  checkCudaErrors(cudaMemcpy(m_d_states, m_states.get(), m_nBlocks * sizeof(State), cudaMemcpyHostToDevice));
  processBatchBlockDevice<<<m_nBlocks, 32>>>(m_d_states, m_singleBlockSize / 8);
#ifndef NDEBUG
  checkCudaErrors(cudaDeviceSynchronize());
#endif
}
