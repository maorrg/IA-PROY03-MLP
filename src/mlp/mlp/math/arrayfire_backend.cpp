#ifdef MLP_USE_ARRAYFIRE_BACKEND

#include <mlp/math/arrayfire_backend.h>

using namespace mlp::math;

static af::randomEngine engine = af::getDefaultRandomEngine();

void mlp::math::set_seed (unsigned long long int seed) { engine.setSeed(seed); }
Matrix mlp::math::zeros (size_t n, size_t m) { return af::constant(0, n, m); }
std::ostream& operator << (std::ostream& os, const Matrix& mat) {
    const char* res = af::toString("", mat);
    os << res;
    af::freeHost(res);
    return os;
}

#endif