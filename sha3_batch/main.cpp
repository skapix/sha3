#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <CLI/CLI.hpp>
#include "util.h"
#include "sha3_cpu.h"
#include "sha3_gpu.h"

namespace
{

std::optional<std::string> readFile(const std::string &filename)
{
  std::ifstream f(filename, std::ios::binary);
  if (!f.is_open())
  {
    return {};
  }
  f.seekg(0, std::ios::end);
  auto length = f.tellg();
  f.seekg(0, std::ios::beg);
  std::string s(length, '\0');
  f.read(s.data(), length);
  return s;
}

template<typename T>
void doCalculation(const std::vector<std::string> &files, const size_t digestSize, const size_t rawBatchSize)
{
  T sha(digestSize);

  // Calculate optimal batch size.
  size_t batchSize = rawBatchSize / sha.batchSize() * sha.batchSize();

  for (size_t i = 0; i < files.size();)
  {

    std::vector<std::string> datas;
    datas.reserve(batchSize);

    std::vector<std::string> names;
    names.reserve(batchSize);

    for (; datas.size() < batchSize && i < files.size(); ++i)
    {
      auto data = readFile(files[i]);
      if (!data.has_value())
      {
        std::cerr << "Unable to open file " << files[i] << std::endl;
        continue;
      }
      names.push_back(files[i]);
      datas.push_back(std::move(data.value()));
    }


    auto args = prepareArgs(datas);
    assert(args.size() == datas.size());
    auto results = sha.calculate(args);
    assert(results.size() == args.size());

    for (size_t j = 0; j < results.size(); ++j)
    {
      std::cout << names[j] << " " << toString(results[j]) << std::endl;
    }
  }
}

} // namespace

int main(int argc, const char *argv[])
{
  size_t digestSize = 512;
  size_t batchSize = 64;
  std::vector<std::string> inputFiles;
  std::vector<std::string> excludeFiles;
  bool isCpu = false;

  CLI::App app("SHA3 hash calculation");
  app.add_set("-d,--digest", digestSize, {224, 256, 384, 512}, "Digest length", true);
  app.add_option("-e,--exclude", excludeFiles, "Exclude files");
  app.add_option("inputs", inputFiles, "Files to calculate SHA3")->check(CLI::ExistingFile)->required();
  app.add_flag("-c,--cpu", isCpu, "Calculate SHA3 hash usign cpu"); // currently unsupported
  app.add_option("-b,--batch-size", batchSize, "Maximum size of batch", true)
      ->check(CLI::Range(size_t(1), std::numeric_limits<size_t>::max()));

  CLI11_PARSE(app, argc, argv);

  // Remove exclude files
  std::set<std::string> includes(inputFiles.begin(), inputFiles.end());
  std::set<std::string> excludes(excludeFiles.begin(), excludeFiles.end());
  inputFiles.clear();
  std::set_difference(includes.begin(), includes.end(), excludes.begin(), excludes.end(),
                      std::back_inserter(inputFiles));

  if (isCpu)
  {
    doCalculation<SHA3_cpu_batch>(inputFiles, digestSize, batchSize);
  }
  else
  {
    doCalculation<SHA3_gpu_batch>(inputFiles, digestSize, batchSize);
  }
}
