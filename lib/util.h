#include <string>
#include <vector>

std::string toString(const std::vector<uint8_t> &res);

std::ostream &operator<<(std::ostream &os, const std::vector<uint8_t> &res);

std::vector<std::vector<uint8_t>> prepareResult(size_t size, size_t digestSize);

std::vector<std::pair<const uint8_t *, size_t>> prepareArgs(const std::vector<std::string> &data);

std::vector<std::pair<const uint8_t *, size_t>> prepareArgs(const std::vector<std::vector<uint8_t>> &data);
