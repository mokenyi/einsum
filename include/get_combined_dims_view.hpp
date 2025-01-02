#ifndef GET_COMBINED_DIMS_VIEW_HPP
#define GET_COMBINED_DIMS_VIEW_HPP
#include "xtensor/xtensor.hpp"
#include "xtensor/xstrided_view.hpp"
#include "has_repeated_labels.hpp"
#include <iostream>

template<typename I, typename E>
typename std::enable_if<
  has_repeated_labels<
    I::start_first,
    I::start_second,
    typename I::tuple_type
  >::value,
  xt::xstrided_view<xt::xclosure_t<E const&>,typename E::shape_type,xt::layout_type::dynamic>
>::type get_combined_dims_view(xt::xexpression<E> const& x) {
  std::cout << "has repeated label" << std::endl;
  auto sh = x.derived_cast().shape();
  auto st = x.derived_cast().strides();
  return xt::strided_view(x.derived_cast(), std::move(sh), std::move(st), 0LU, xt::layout_type::dynamic);
}

template<typename I, typename E>
typename std::enable_if<
  !has_repeated_labels<
    I::start_first,
    I::start_second,
    typename I::tuple_type
  >::value,
  xt::xexpression<E> const&
>::type get_combined_dims_view(xt::xexpression<E> const& x) {
  std::cout << "no repeated label" << std::endl;
  return x;
}

#endif // GET_COMBINED_DIMS_VIEW_HPP
