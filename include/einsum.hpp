#ifndef EINSUM_HPP
#define EINSUM_HPP

#include "xtensor/xarray.hpp"
#include <type_traits>

template<typename O, typename... Ss>
struct einsum {
  template<typename... Ts>
  auto eval(xt::xexpression<Ts> const&... op_in) -> xt::xarray<typename std::common_type<Ts::value_type...>::type>;
  int ellipsis;
  einsum(int e = static_cast<int>('_'));
}

#define // EINSUM_HPP
