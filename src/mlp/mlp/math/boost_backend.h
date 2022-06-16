//
// Created by Esteban on 6/15/2022.
//

#ifndef UNTITLED21_BOOST_BACKEND_H
#define UNTITLED21_BOOST_BACKEND_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>

template<class E1, class E2>
inline auto operator * (
    const boost::numeric::ublas::matrix_expression<E1>& a,
    const boost::numeric::ublas::matrix_expression<E2>& b
) {
    return boost::numeric::ublas::element_prod(a, b);
}

namespace mlp::math {
    using namespace boost::numeric::ublas;
    using Matrix = matrix<double>;
    using Vector = vector<double>;

    size_t size (const Matrix& m, size_t dim);
    Vector constant (size_t n, double value);
    Matrix constant (size_t n, size_t m, double value);
    Matrix zeros (size_t n, size_t m);
    Matrix ones (size_t n, size_t m);
    Matrix randu (size_t n, size_t m);
    void setSeed (unsigned seed);
    template<class ... Args>
    inline void eval (const Args& ...) {};
    void setDevice (int);
    void info ();
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
