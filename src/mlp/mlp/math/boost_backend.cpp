#ifdef MLP_USE_BOOST_BACKEND

#include <mlp/math/boost_backend.h>


static auto random_source = std::random_device{};
static auto random_engine = std::default_random_engine{ random_source() };
static auto dist = std::uniform_real_distribution<double>(0, 1);
static std::normal_distribution<double> ndist(0.0, 1.0);
static double random_unif () { return dist(random_engine); }
static double random_norm () { return ndist(random_engine); }

using namespace mlp::math;

size_t mlp::math::size (const Matrix& m, size_t dim) { return dim == 0 ? m.size1() : m.size2(); }
Vector mlp::math::constant (size_t n, double value) { return scalar_vector<double>(n, value); }
Matrix mlp::math::constant (size_t n, size_t m, double value) { return scalar_matrix<double>(n, m, value); }
Matrix mlp::math::ones (size_t n, size_t m) { return constant(n, m, 1); }
Matrix mlp::math::zeros (size_t n, size_t m) { return constant(n, m, 0); }
Matrix mlp::math::randu (size_t n, size_t m) {
    Matrix mat{ n, m };
    std::generate(mat.data().begin(), mat.data().end(), random_unif);
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
vector<int> mlp::math::shuffle_idx (int size) {
    vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), random_engine);
    return idx;
}

Matrix mlp::math::randn (size_t n, size_t m) {
    Matrix mat{ n, m };
    std::generate(mat.data().begin(), mat.data().end(), random_norm);
    return mat;
}

std::ostream& operator << (std::ostream& os, const boost::numeric::ublas::matrix<double>& mat) {
    const auto p_precision = os.precision();
    os << "[" << mlp::math::size(mat, 0) << ", " << mlp::math::size(mat, 1) << "]:\n";
    for (size_t i = 0; i < mat.size1(); ++i) {
        for (size_t j = 0; j < mat.size2(); ++j) {
            os << std::setprecision(6) << std::showpos << std::fixed << mat(i, j) << "\t";
        }
        os << std::endl;
    }
    os << std::setprecision(p_precision);
    return os;
}


#endif // MLP_USE_BOOST_BACKEND

