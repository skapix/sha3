#include <iostream>
#include <fstream>
#include <vector>
#include <optional>
#include <charconv>
#include <chrono>
#include <cstring>
#include <cassert>
#include <map>
#include "util.h"
#include <algorithm>
#include "sha3_cpu.h"
#include "sha3_gpu.h"
#include <CLI/CLI.hpp>

namespace
{

const std::string g_singleSubcommand = "single";
const std::string g_batchSubcommand = "batch";

const size_t g_kb = 1024;
const size_t g_mb = g_kb * 1024;
const size_t g_gb = g_mb * 1024;

enum class RunType
{
  Cpu,
  Gpu
};

std::string toString(RunType t)
{
  if (t == RunType::Cpu)
  {
    return "Cpu";
  }
  if (t == RunType::Gpu)
  {
    return "Gpu";
  }
  return "";
}

void writeHeader(std::ostream &out, const std::vector<size_t> &sizes)
{
  out << "Type";
  const char *sizeSuffix[] = {"b", "kb", "mb", "gb"};
  for (size_t sz : sizes)
  {
    out << ",";

    for (size_t i = 0; i < 3; ++i)
    {
      if (sz < 1024)
      {
        out << sz << sizeSuffix[i];
        break;
      }
      sz /= 1024;
    }
  }
  out << std::endl;
}

void writeBatchHeader(std::ostream &out, const std::vector<size_t> &sizes)
{
  out << "Batch size,";
  writeHeader(out, sizes);
}

template<typename T>
std::vector<uint8_t> measureSingleSha3(size_t digest, const uint8_t *data, const size_t size, std::ostream &out)
{
  T sha3(digest);

  // warm-up
  size_t testSize = std::min(size, g_mb);
  sha3.add(data, testSize);
  sha3.digest();
  sha3.init();

  // start test
  auto p1 = std::chrono::high_resolution_clock::now();
  sha3.add(data, size);
  auto result = sha3.digest();
  auto p2 = std::chrono::high_resolution_clock::now();
  double diff_ms = std::chrono::duration<double, std::milli>(p2 - p1).count();
  out << diff_ms << std::flush;

  return result;
}

template<typename T>
std::vector<std::vector<uint8_t>>
    measureBatchSha3(T &sha3Batch, const std::vector<std::pair<const uint8_t *, size_t>> &datas, std::ostream &out)
{
  // warm-up
  std::vector<std::pair<const uint8_t *, size_t>> trimmed;
  for (auto &data : datas)
  {
    trimmed.push_back({data.first, std::min(g_mb, data.second)});
  }

  sha3Batch.calculate(trimmed);

  // start test
  auto p1 = std::chrono::high_resolution_clock::now();
  auto result = sha3Batch.calculate(datas);
  auto p2 = std::chrono::high_resolution_clock::now();
  double diff_ms = std::chrono::duration<double, std::milli>(p2 - p1).count();
  out << diff_ms << std::flush;

  return result;
}

bool parseUnsigned(const std::string &in, size_t &result)
{
  size_t value;
  auto res = std::from_chars(in.data(), in.data() + in.size(), value);
  if (res.ec != std::errc::invalid_argument && res.ec != std::errc::result_out_of_range)
  {
    result = value;
    return true;
  }
  return false;
}


void runSingleTest(std::ostream &out, const std::size_t digestSize, const std::vector<size_t> &sizes,
                   const std::vector<RunType> &runTypes)
{
  writeHeader(out, sizes);

  size_t max = *std::max_element(sizes.begin(), sizes.end());
  std::vector<uint8_t> data(max);
  std::generate(data.begin(), data.end(), rand);

  for (auto type : runTypes)
  {
    out << toString(type);

    for (auto size : sizes)
    {
      out << ",";
      std::vector<uint8_t> result;

      if (type == RunType::Cpu)
      {
        result = measureSingleSha3<SHA3_cpu>(digestSize, data.data(), size, out);
      }
      else if (type == RunType::Gpu)
      {
        result = measureSingleSha3<SHA3_gpu>(digestSize, data.data(), size, out);
      }
      (void)result;
    }
    out << std::endl;
  }
}

void runBatchTest(std::ostream &out, const std::size_t digestSize, const std::vector<size_t> &sizes,
                  const size_t batchSize, const std::vector<RunType> &runTypes, bool tweakBatchSize)
{
  SHA3_cpu_batch cpu(digestSize);
  SHA3_gpu_batch gpu(digestSize);
  auto getBatchSize = [&](RunType t) {
    if (!tweakBatchSize)
    {
      return batchSize;
    }
    switch (t)
    {
    case RunType::Cpu:
      return batchSize / cpu.batchSize() * cpu.batchSize();
    case RunType::Gpu:
      return batchSize / gpu.batchSize() * gpu.batchSize();
    default:
      assert(false);
      return size_t(0);
    }
  };

  writeBatchHeader(out, sizes);

  size_t max = *std::max_element(sizes.begin(), sizes.end());
  std::vector<std::vector<uint8_t>> datas;
  for (size_t i = 0; i < batchSize; ++i)
  {
    std::vector<uint8_t> data(max);
    std::generate(data.begin(), data.end(), rand);
    datas.push_back(std::move(data));
  }

  auto prepared = prepareArgs(datas);

  for (auto type : runTypes)
  {
    size_t realBatchSize = getBatchSize(type);
    out << realBatchSize << ',' << toString(type);

    for (auto size : sizes)
    {
      auto local = prepared;
      local.resize(realBatchSize);
      for (auto &it : local)
      {
        it.second = size;
      }

      out << ",";
      std::vector<std::vector<uint8_t>> result;

      if (type == RunType::Cpu)
      {
        result = measureBatchSha3<SHA3_cpu_batch>(cpu, local, out);
      }
      else if (type == RunType::Gpu)
      {
        result = measureBatchSha3<SHA3_gpu_batch>(gpu, local, out);
      }
      (void)result;
    }
    out << std::endl;
  }
}

} // namespace

// Global TODO: global README and License

int main(int argc, const char *argv[])
{
#ifndef NDEBUG
  std::cerr << "Warning: code is run in debug mode. Results may be incorrect." << std::endl;
#endif

  std::vector<size_t> singleSizes = {1 * g_mb, 10 * g_mb, 20 * g_mb, 50 * g_mb};
  std::vector<size_t> batchSizes = {1 * g_mb, 10 * g_mb};
  size_t digestSize = 512;
  size_t batchSize = 64; // batch only;
  bool noBatchCorrection = false; // batch only;
  std::string outFilename;

  size_t nCpu = 1;
  size_t nGpu = 1;

  auto addCommonCli = [&](CLI::App *app, std::vector<size_t> &sizes, bool batch) {
    static const std::set<size_t> digestOptions = {224, 256, 384, 512};
    if (batch)
    {
      app->add_option("-b,--batch", batchSize, "", true);
    }
    app->add_option("-c,--cpu", nCpu, "Run count of cpu benchmark to run", true);
    app->add_set("-d,--digest", digestSize, digestOptions, "Digest length", true);
    app->add_option("-g,--gpu", nGpu, "Run count of gpu benchmark to run", true);
    if (batch)
    {
      app->add_flag("-n,--no-batch-corection", noBatchCorrection, "Batch correction is used to maximize performance");
    }
    app->add_option("-o,--output-file", outFilename, "Output file (STDIN if not specified)");
    app->add_option("-s,--sizes", sizes, "Data sizes to benchark", true);
  };

  CLI::App app("SSH benchmark");
  auto single = app.add_subcommand(g_singleSubcommand, "benchmark of 'single-threaded' sha3");
  addCommonCli(single, singleSizes, false);

  auto batch = app.add_subcommand(g_batchSubcommand, "benchmark of batch sha3");
  addCommonCli(batch, batchSizes, true);

  app.require_subcommand(1);

  CLI11_PARSE(app, argc, argv);

  assert(app.get_subcommands().size() == 1);
  std::string subcommand = app.get_subcommands().front()->get_name();


  std::ofstream of;
  if (!outFilename.empty())
  {
    of.open(outFilename);
    if (!of.is_open())
    {
      std::cerr << "Unable to open file " << outFilename << std::endl;
      exit(EXIT_FAILURE);
    }
  }


  std::ostream &out = of.is_open() ? of : std::cout;

  std::vector<RunType> runTypes(nCpu, RunType::Cpu);
  runTypes.insert(runTypes.end(), nGpu, RunType::Gpu);

  if (subcommand == g_batchSubcommand)
  {
    runBatchTest(out, digestSize, batchSizes, batchSize, runTypes, !noBatchCorrection);
  }
  else if (subcommand == g_singleSubcommand)
  {
    runSingleTest(out, digestSize, singleSizes, runTypes);
  }
  else
  {
    assert(false);
  }
}
