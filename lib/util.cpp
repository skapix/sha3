#include "util.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

std::string toString(const std::vector<uint8_t> &res)
{
  std::ostringstream os;
  for (uint8_t it : res)
  {
    os << std::hex << std::setw(2) << std::setfill('0') << +it;
  }
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const std::vector<uint8_t> &res)
{
  os << toString(res);
  return os;
}

std::vector<std::vector<uint8_t>> prepareResult(size_t size, size_t digestSize)
{
  std::vector<std::vector<uint8_t>> result;
  result.reserve(size);
  for (size_t i = 0; i < size; ++i)
  {
    result.emplace_back(digestSize / 8);
  }
  return result;
}

std::vector<std::pair<const uint8_t *, size_t>> prepareArgs(const std::vector<std::string> &data)
{
  std::vector<std::pair<const uint8_t *, size_t>> result;
  result.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(result),
                 [](const std::string &str) -> std::pair<const uint8_t *, size_t> {
                   return {reinterpret_cast<const uint8_t *>(str.data()), str.size()};
                 });
  return result;
}

std::vector<std::pair<const uint8_t *, size_t>> prepareArgs(const std::vector<std::vector<uint8_t>> &datas)
{
  std::vector<std::pair<const uint8_t *, size_t>> result;
  result.reserve(datas.size());
  std::transform(datas.begin(), datas.end(), std::back_inserter(result),
                 [](const std::vector<uint8_t> &data) -> std::pair<const uint8_t *, size_t> {
                   return {data.data(), data.size()};
                 });
  return result;
}
