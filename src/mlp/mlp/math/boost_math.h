//
// Created by Esteban on 6/15/2022.
//

#ifndef UNTITLED21_BOOST_MATH_H
#define UNTITLED21_BOOST_MATH_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>

namespace mlp::math {
    namespace ub = boost::numeric::ublas;
    using Matrix = ub::matrix<double>;
    using Vector = ub::vector<double>;

    template<class E1, class E2>
    inline auto operator * (const ub::matrix_expression<E1>& a, const ub::matrix_expression<E2>& b) {
        return ub::element_prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const ub::matrix_expression<E1>& a, const ub::matrix_expression<E2>& b) {
        return boost::numeric::ublas::prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const ub::matrix_expression<E1>& a, const ub::vector_expression<E2>& b) {
        return boost::numeric::ublas::prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const ub::vector_expression<E1>& a, const ub::vector_expression<E2>& b) {
        return boost::numeric::ublas::prod(a, b);
    }

    template<class E1, class E2>
    inline auto operator % (const ub::vector_expression<E1>& a, const ub::matrix_expression<E2>& b) {
        return boost::numeric::ublas::prod(a, b);
    }

    template<class M>
    inline auto col (M& a, size_t i) { return ub::column(a, i); }

    template<class M>
    inline auto col (const M& a, size_t i) { return ub::column(a, i); }

    template<class M>
    inline auto row (M& a, size_t i) { return ub::row(a, i); }

    template<class M>
    inline auto row (const M& a, size_t i) { return ub::row(a, i); }

    template<class E>
    inline auto trans (const ub::matrix_expression<E>& a) { return ub::trans(a); }

    inline auto full (size_t n, double value) { return ub::scalar_vector<double>(n, value); }

    inline auto full (size_t m, size_t n, double value) { return ub::scalar_matrix<double>(m, n, value); }

    template <class E>
    inline auto sum (const ub::vector_expression<E>& a) { return ub::sum(a); }
};

#define MLP_BOOST_MATH_MAKE_FUNCTOR(NAME, OP) \
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
    typename mlp::math::ub::matrix_unary1_traits<E, functor::NAME<typename E::value_type>>::result_type \
    NAME (const mlp::math::ub::matrix_expression<E>& e) { \
        typedef typename mlp::math::ub::matrix_unary1_traits<E, functor::NAME<typename E::value_type>>::expression_type expression_type; \
        return expression_type(e()); \
    } \
    template<class E> \
    BOOST_UBLAS_INLINE \
    typename mlp::math::ub::vector_unary_traits<E, functor::NAME<typename E::value_type>>::result_type \
    NAME (const mlp::math::ub::vector_expression<E>& e) { \
        typedef typename mlp::math::ub::vector_unary_traits<E, functor::NAME<typename E::value_type>>::expression_type expression_type; \
        return expression_type(e()); \
    } \
} \



#endif //UNTITLED21_BOOST_MATH_H
