#ifndef EINSUM_HPP
#define EINSUM_HPP

#include "xtensor/xarray.hpp"
#include <type_traits>

template<typename... Ss>
struct einsum {
  template<typename O, typename... Ts>
  auto eval(xt::xexpression<Ts> const&... op_in) -> xt::xarray<typename std::common_type<typename Ts::value_type...>::type>;
  int ellipsis;
  einsum(int e = static_cast<int>('_'));
};

#include "impl/einsum.tpp"

#endif // EINSUM_HPP
