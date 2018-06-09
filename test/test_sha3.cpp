#include "gtest/gtest.h"
#include "sha3_gpu.h"
#include "sha3_cpu.h"
#include "util.h"
#include <string>
#include <vector>

namespace
{

template<typename T>
void partialTest(int size, const std::string &value, const std::string &expected, size_t split)
{
  assert(split <= value.size());
  T sha(size);
  sha.add(reinterpret_cast<const uint8_t *>(value.data()), split);
  sha.add(reinterpret_cast<const uint8_t *>(value.data()) + split, value.size() - split);
  std::vector<uint8_t> result = sha.digest();
  auto var = toString(result);
  EXPECT_EQ(expected, var);
}

const char *g_story = "Little Red Riding Hood ran away from an angry gray wolf on motobyke. He was very hungry and ate "
                      "her breakfast. She was driving the first byke in the history as it happened in 1084. "
                      "Story began... Read the continuation in \"Fictitious Little Red Riding Hood Stories\"";

using TestCaseData = std::vector<std::pair<std::string, std::string>>;
TestCaseData g_224 = {
    {"", "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7"},
    {"123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097"},
    {"sha3!@#", "ba3b6686ac5077da6d21aca60f0a9f52bc606d2fba40693cfbe2007b"},
    {g_story, "133b9f17b93af273ffc00a07b1b18da15b4ebe8a74ed302f6d4540e6"},
};

TestCaseData g_256 = {
    {"", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"},
    {"123", "a03ab19b866fc585b5cb1812a2f63ca861e7e7643ee5d43fd7106b623725fd67"},
    {"sha3!@#", "05f64ea16f3ad32d88927e00049017458c154c5d7b35d32c315f458b121eb4f7"},
    {g_story, "93109f7a3a19b7da48ef32e4ba61e33c8ecb97d905797cd7a2443ffb8ae03db1"},
};

TestCaseData g_384 = {
    {"", "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004"},
    {"123", "9bd942d1678a25d029b114306f5e1dae49fe8abeeacd03cfab0f156aa2e363c988b1c12803d4a8c9ba38fdc873e5f007"},
    {"sha3!@#", "11379c25373626daca9d16c368eac54a0ffb25a4fd1ea20063bba91b5a99f41fd36c8a9b2285f1173bed391221caafa7"},
    {g_story, "f4bb32363ba3751fabdd524efcfcfd7e5f817af51e96347414661b0082c5eb40f41ee46bebd7ef024e0f59d33c013c99"},
};

TestCaseData g_512 = {
    {"", "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c"
         "3ac558f500199d95b6d3e301758586281dcd26"},
    {"123", "48c8947f69c054a5caa934674ce8881d02bb18fb59d5a63eeaddff735b0e9801e87294783281ae49fc8287a"
            "0fd86779b27d7972d3e84f0fa0d826d7cb67dfefc"},
    {"sha3!@#", "bf73d68f1ad743ff82dbd61dd2f51d68532cad9f1bb177b448aaf34bcebb3420211cbe992b5a4f04c05"
                "5cfcd5c3801a04616249a933e976685dcd3ab030afd98"},
    {g_story, "fda9b20daea98a4bdb422adda990af4ff79212d73997cb6745daa1150ca9e2012f80cb54b41436fc30990"
              "4e9c07af58daefce4f645fa69649c6c1398e22951b8"},
};

std::string g_10MbZeroesDigest512 = "4d0287eff3cc77d3d570c06efe9c94dbd848f9a935f2c50fe68bd7c2ec70cb58565aa02778fc9bd890"
                                    "f0497e2fed03201582778f495db8d2eecc30225ea1643b";
template<typename T>
class TestCase {
public:
  TestCase(size_t digestSize)

    : t(digestSize)
  {}

  void doTest(TestCaseData::const_iterator begin, TestCaseData::const_iterator end)
  {
    std::string scopeName = std::string("Type: ") + typeid(t).name();
    SCOPED_TRACE(scopeName.c_str());
    for (auto it = begin; it != end; ++it)
    {
      std::string scopeCalc = "Calculation " + it->first;
      SCOPED_TRACE(scopeCalc.c_str());
      if (it != begin)
      {
        t.init();
      }
      t.add(reinterpret_cast<const uint8_t *>(it->first.data()), it->first.size());
      auto result = toString(t.digest());
      EXPECT_EQ(it->second, result);
    }
  }

  void doBatchTest(TestCaseData::const_iterator begin, TestCaseData::const_iterator end)
  {
    std::vector<std::pair<const uint8_t *, size_t>> data;
    std::vector<std::string> expected;
    std::for_each(begin, end, [&data, &expected](const std::pair<std::string, std::string> &pr) {
      data.push_back({(const uint8_t *)pr.first.data(), pr.first.size()});
      expected.push_back(pr.second);
    });

    auto almostResult = t.calculate(data);
    std::vector<std::string> result;
    std::transform(almostResult.begin(), almostResult.end(), std::back_inserter(result), toString);

    EXPECT_EQ(expected, result);
  }

private:
  T t;
};

template<typename T>
void largeArrayBatchTest()
{
  auto vec = g_224;
  for (size_t i = 0; i < 100; ++i)
  {
    vec.insert(vec.end(), g_224.begin(), g_224.end());
  }

  TestCase<T> cpu(224);
  cpu.doBatchTest(vec.begin(), vec.end());
}

} // namespace

TEST(sha3_checks_gpu, partial)
{
  partialTest<SHA3_gpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 0);
  partialTest<SHA3_gpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 1);
  partialTest<SHA3_gpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 2);
  partialTest<SHA3_gpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 3);
}

TEST(sha3_checks_cpu, partial)
{
  partialTest<SHA3_cpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 0);
  partialTest<SHA3_cpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 1);
  partialTest<SHA3_cpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 2);
  partialTest<SHA3_cpu>(224, "123", "602bdc204140db016bee5374895e5568ce422fabe17e064061d80097", 3);
}

TEST(sha3_checks_cpu, common_224)
{
  TestCase<SHA3_cpu> gpu(224);
  gpu.doTest(g_224.begin(), g_224.end());
}

TEST(sha3_checks_cpu, common_256)
{
  TestCase<SHA3_cpu> gpu(256);
  gpu.doTest(g_256.begin(), g_256.end());
}

TEST(sha3_checks_cpu, common_384)
{
  TestCase<SHA3_cpu> gpu(384);
  gpu.doTest(g_384.begin(), g_384.end());
}


TEST(sha3_checks_cpu, common_512)
{
  TestCase<SHA3_cpu> gpu(512);
  gpu.doTest(g_512.begin(), g_512.end());
}

TEST(sha3_checks_gpu, common_224)
{
  TestCase<SHA3_gpu> gpu(224);
  gpu.doTest(g_224.begin(), g_224.end());
}

TEST(sha3_checks_gpu, common_256)
{
  TestCase<SHA3_gpu> gpu(256);
  gpu.doTest(g_256.begin(), g_256.end());
}

TEST(sha3_checks_gpu, common_384)
{
  TestCase<SHA3_gpu> gpu(384);
  gpu.doTest(g_384.begin(), g_384.end());
}

TEST(sha3_checks_gpu, common_512)
{
  TestCase<SHA3_gpu> gpu(512);
  gpu.doTest(g_512.begin(), g_512.end());
}

TEST(sha3_batch_checks_cpu, large_224) { largeArrayBatchTest<SHA3_cpu_batch>(); }

TEST(sha3_batch_checks_gpu, large_224) { largeArrayBatchTest<SHA3_gpu_batch>(); }

TEST(sha3_batch_checks_cpu, common_224)
{
  TestCase<SHA3_cpu_batch> cpu(224);
  cpu.doBatchTest(g_224.begin(), g_224.end());
}

TEST(sha3_batch_checks_cpu, common_256)
{
  TestCase<SHA3_cpu_batch> cpu(256);
  cpu.doBatchTest(g_256.begin(), g_256.end());
}

TEST(sha3_batch_checks_cpu, common_384)
{
  TestCase<SHA3_cpu_batch> cpu(384);
  cpu.doBatchTest(g_384.begin(), g_384.end());
}

TEST(sha3_batch_checks_cpu, common_512)
{
  TestCase<SHA3_cpu_batch> cpu(512);
  cpu.doBatchTest(g_512.begin(), g_512.end());
}


TEST(sha3_batch_checks_gpu, common_224)
{
  TestCase<SHA3_gpu_batch> gpu(224);
  gpu.doBatchTest(g_224.begin(), g_224.end());
}

TEST(sha3_batch_checks_gpu, common_256)
{
  TestCase<SHA3_gpu_batch> gpu(256);
  gpu.doBatchTest(g_256.begin(), g_256.end());
}

TEST(sha3_batch_checks_gpu, common_384)
{
  TestCase<SHA3_gpu_batch> gpu(384);
  gpu.doBatchTest(g_384.begin(), g_384.end());
}

TEST(sha3_batch_checks_gpu, common_512)
{
  TestCase<SHA3_gpu_batch> gpu(512);
  gpu.doBatchTest(g_512.begin(), g_512.end());
}

TEST(sha3, large_file)
{
  const size_t mb = 1024 * 1024;
  std::vector<uint8_t> data(10 * mb);

  SHA3_cpu c(512);
  c.add(data.data(), data.size());
  auto resultCpu = c.digest();
  ASSERT_EQ(g_10MbZeroesDigest512, toString(resultCpu));

  SHA3_gpu g(512);
  g.add(data.data(), data.size());
  auto resultGpu = g.digest();
  ASSERT_EQ(resultCpu, resultGpu);

  std::vector<std::pair<const uint8_t *, size_t>> batchArg = {{data.data(), data.size()}};

  SHA3_cpu_batch cb(512);
  auto resultCpuBatch = cb.calculate(batchArg).front();
  ASSERT_EQ(resultCpu, resultCpuBatch);

  SHA3_gpu_batch gb(512);
  auto resultGpuBatch = gb.calculate(batchArg).front();
  ASSERT_EQ(resultCpu, resultGpuBatch);
}
