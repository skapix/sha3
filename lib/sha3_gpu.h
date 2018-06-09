#pragma once
#include <cstdint>
#include <vector>
#include <memory>

class SHA3_gpu {
public:
  SHA3_gpu(size_t block);
  ~SHA3_gpu();

  void init();
  void add(const uint8_t *data, size_t sz);

  std::vector<uint8_t> digest();

private:
  void processBlock(size_t bufSize);

private:
  size_t m_digestSize;

  size_t m_singleBufSz = 0;
  static const size_t m_nBuffers = 128;
  uint8_t *m_d_blockBuffers = nullptr; // Device memory block.
  size_t m_bufferOffset = 0;

  uint64_t *m_d_A; // Device state array of size 25.

  bool m_finished = false;
};


class SHA3_gpu_batch {
public:
  using Digest = std::vector<uint8_t>;

  SHA3_gpu_batch(size_t block);
  ~SHA3_gpu_batch();

  // Note, it's better for array to be sorted by descending size.
  std::vector<Digest> calculate(const std::vector<std::pair<const uint8_t *, size_t>> &datas);
  size_t batchSize() const { return m_nBlocks; }

  struct State;

private:
  std::vector<Digest> prepareResult(size_t size);
  void launchKernel();

private:
  size_t m_digestSize;
  size_t m_singleBlockSize;
  size_t m_bufferSize;
  size_t m_nBlocks;
  std::unique_ptr<State[]> m_states;
  State *m_d_states;
  static const size_t m_maxBuffers = 128;
};
