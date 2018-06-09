#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <CLI/CLI.hpp>
#include "util.h"
#include "sha3_cpu.h"
#include "sha3_gpu.h"

template<typename T>
std::vector<uint8_t> doCalculation(std::istream &is, size_t digestSize, size_t bufSize)
{
  T s(digestSize);

  std::vector<uint8_t> data;
  data.resize(bufSize);

  while (true)
  {
    is.read(reinterpret_cast<char *>(data.data()), data.size());
    std::streamsize nread = is.gcount();
    assert(nread >= 0);
    if (nread == 0)
    {
      break;
    }
    s.add(data.data(), static_cast<size_t>(nread));
  }

  return s.digest();
}

int main(int argc, const char *argv[])
{
  size_t digestSize = 512;
  std::string inputFile;
  bool isGpu = false;

  CLI::App app("SHA3 hash calculation");
  app.add_set("-d,--digest", digestSize, {224, 256, 384, 512}, "Digest length", true);
  app.add_option("input", inputFile, "File to calculate SHA3")->check(CLI::ExistingFile);
  app.add_flag("-g,--gpu", isGpu, "Calculate SHA3 hash usign gpu");

  CLI11_PARSE(app, argc, argv);

  std::ifstream f(inputFile, std::ifstream::binary);

  if (!f.is_open())
  {
    std::cerr << "Can't open file " << argv[1] << std::endl;
    return 1;
  }

  size_t size = 1024 * 1024;
  std::vector<uint8_t> digest;
  if (isGpu)
  {
    digest = doCalculation<SHA3_gpu>(f, digestSize, size);
  }
  else
  {
    digest = doCalculation<SHA3_cpu>(f, digestSize, size);
  }

  std::cout << digest << std::endl;
}
