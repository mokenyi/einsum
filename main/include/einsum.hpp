#ifndef EINSUM_HPP
#define EINSUM_HPP

#include "xtensor/xarray.hpp"
#include <type_traits>

/**
 * Wrapper struct for the eval() method, which evaluates an Einstein summation
 * over a set of operands.
 * Ss: subscript types
 */
template<typename... Ss>
struct einsum {
  /**
   * Evaluate an Einstein summation over a set of operands.
   * O: output subscript type (set to implicit_out for implicit mode)
   * Ts: xexpression types
   * ellipsis: the value that denotes a set of broadcast dimensions (equivalent
   *   to '...' in numpy.einsum()
   */
  template<typename O, typename... Ts>
  auto eval(xt::xexpression<Ts> const&... op_in) -> xt::xarray<typename std::common_type<typename Ts::value_type...>::type>;
  int ellipsis;
  einsum(int e = 0);
};

#include "impl/einsum.tpp"

#endif // EINSUM_HPP
