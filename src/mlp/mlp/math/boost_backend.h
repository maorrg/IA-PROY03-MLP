//
// Created by Esteban on 6/15/2022.
//

#ifndef UNTITLED21_BOOST_BACKEND_H
#define UNTITLED21_BOOST_BACKEND_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>

namespace mlp::math {
    using namespace boost::numeric::ublas;
    using Matrix = matrix<double>;

    template<class E1, class E2>
    inline auto operator * (const matrix_expression<E1>& a, const matrix_expression<E2>& b) {
        return element_prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const matrix_expression<E1>& a, const matrix_expression<E2>& b) {
        return prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const matrix_expression<E1>& a, const vector_expression<E2>& b) {
        return prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const vector_expression<E1>& a, const vector_expression<E2>& b) {
        return prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const vector_expression<E1>& a, const matrix_expression<E2>& b) {
        return prod(a, b);
    }

    inline auto full (size_t n, double value) { return scalar_vector<double>(n, value); }

    inline auto full (size_t m, size_t n, double value) { return scalar_matrix<double>(m, n, value); }

    static auto random_source = std::random_device{};
    static auto random_engine = std::default_random_engine{ random_source() };
    static auto dist = std::uniform_real_distribution<double>(0, 1);

    inline auto random () {
        return dist(random_engine);
    };

    inline auto randu(size_t n, size_t m) {
        Matrix mat{n, m};
        std::generate(mat.data().begin(), mat.data().end(), random);
        return mat;
    }

    inline auto set_seed(unsigned seed) {
        random_engine = std::default_random_engine{ seed };
    }
};

#define MLP_MATH_MAKE_FUNCTOR(NAME, OP) \
namespace functor { \
    template <class T> \
    struct NAME { \
        typedef T value_type; \
        typedef T result_type; \
        static result_type apply(const value_type& x) { return OP(x); } \
    }; \
} \
namespace mlp::math { \
    template<class E> \
    BOOST_UBLAS_INLINE \
    typename mlp::math::matrix_unary1_traits<E, functor::NAME<typename E::value_type>>::result_type \
    NAME (const mlp::math::matrix_expression<E>& e) { \
        typedef typename mlp::math::matrix_unary1_traits<E, functor::NAME<typename E::value_type>>::expression_type expression_type; \
        return expression_type(e()); \
    } \
    template<class E> \
    BOOST_UBLAS_INLINE \
    typename mlp::math::vector_unary_traits<E, functor::NAME<typename E::value_type>>::result_type \
    NAME (const mlp::math::vector_expression<E>& e) { \
        typedef typename mlp::math::vector_unary_traits<E, functor::NAME<typename E::value_type>>::expression_type expression_type; \
        return expression_type(e()); \
    } \
} \



#endif //UNTITLED21_BOOST_BACKEND_H
