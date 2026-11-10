// Portable host C++ fixture for M2 source-intake testing.
// This is not an Ascend device kernel or a validated A5 binary.
#include <cstddef>

extern "C" void forward(const float* x, float* result, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = x[i] + 1.0f;
    }
}
