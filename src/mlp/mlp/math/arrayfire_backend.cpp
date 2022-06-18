#ifdef MLP_USE_ARRAYFIRE_BACKEND

#include <mlp/math/arrayfire_backend.h>


using namespace mlp::math;

size_t mlp::math::size (const Matrix& m, size_t dim) { return m.dims((int) dim); }
void mlp::math::setSeed (unsigned long long int seed) { af::setSeed(seed); }
Matrix mlp::math::zeros (size_t n, size_t m) { return af::constant(0, (int) n, (int) m); }
Matrix mlp::math::shuffle_idx (size_t size) {
    af::array tmp = af::randu((int) size, 1);
    af::array val, idx;
    af::sort(val, idx, tmp);
    return idx;
}

#undef max
Matrix mlp::math::argmax (const Matrix& mat, size_t dim) {
    af::array res, ignored;
    af::max(ignored, res, mat, (int) dim);
    return res;
}


std::ostream& operator << (std::ostream& os, const Matrix& mat) {
    const char* res = af::toString("", mat);
    os << res;
    af::freeHost(res);
    return os;
}


#endif // MLP_USE_ARRAYFIRE_BACKEND
