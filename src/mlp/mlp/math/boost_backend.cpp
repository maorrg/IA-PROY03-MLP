#ifdef MLP_USE_BOOST_BACKEND

#include <mlp/math/boost_backend.h>


static auto random_source = std::random_device{};
static auto random_engine = std::default_random_engine{ random_source() };
static auto dist = std::uniform_real_distribution<double>(0, 1);
static double random () { return dist(random_engine); }

using namespace mlp::math;

size_t mlp::math::size (const Matrix& m, size_t dim) { return dim == 0 ? m.size1() : m.size2(); }
Vector mlp::math::constant (size_t n, double value) { return scalar_vector<double>(n, value); }
Matrix mlp::math::constant (size_t n, size_t m, double value) { return scalar_matrix<double>(n, m, value); }
Matrix mlp::math::ones (size_t n, size_t m) { return constant(n, m, 1); }
Matrix mlp::math::zeros (size_t n, size_t m) { return constant(n, m, 0); }
Matrix mlp::math::randu (size_t n, size_t m) {
    Matrix mat{ n, m };
    std::generate(mat.data().begin(), mat.data().end(), random);
    return mat;
}
void mlp::math::setSeed (unsigned int seed) {
    random_engine = std::default_random_engine{ seed };
}
void mlp::math::setDevice (int) {
    // no-op
}
void mlp::math::info () {
    // no-op
}
void mlp::math::eval (...) {
    // no-op
}

#endif

