#include <algorithm>
#include <iostream>
#include "einsum_util.hpp"

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N == sizeof...(Ts)-1, size_t>::type get_max_op_ndim(
  std::tuple<Ts...> const& x
) {
  return std::get<N>(x).dimension();
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N < sizeof...(Ts)-1, size_t>::type get_max_op_ndim(
  std::tuple<Ts...> const& x
) {
  return std::max(std::get<N>(x).dimension(), get_max_op_ndim<N+1>(x));
}

template<typename... Ts>
operands_container<Ts...>::operands_container(
  std::tuple<Ts...> const& x,
  op_axes_type const& op_axes
)
: exp(x)
, op_axes(op_axes)
, ndim(op_axes.at(0).size())
{
  op_idx.fill(0LU);
  set_op_sh(op_sh, x);
  set_shape();
}

template<typename... Ts>
void operands_container<Ts...>::set_shape() {
  std::array<size_t,num_ops> i;
  std::iota(i.begin(), i.end(), 0);
  shape.resize(ndim);
  for (size_t j=0; j<ndim; ++j) {
    // Points to first operand which has a non-broadcast axis that maps
    // to dim j of the iterator.
    auto non_broadcast_op_it = std::find_if(
      i.begin(),
      i.end(),
      [this, j](size_t i_p) {
        return op_axes.at(i_p).at(j) != -1;
      }
    );
    size_t const size_dim_j = op_sh
      .at(*non_broadcast_op_it)
      .at(op_axes.at(*non_broadcast_op_it).at(j));

    auto bad_op_it = std::find_if_not(i.begin(), i.end(),
      [this, j, size_dim_j](size_t i_p) {
        return op_axes.at(i_p).at(j) == -1 || op_sh.at(i_p).at(op_axes.at(i_p).at(j)) == size_dim_j;
      });

    if (bad_op_it != i.end()) {
      std::stringstream ss;
      size_t const bad_axis = op_axes.at(*bad_op_it).at(j);
      ss << "For dimension " << j << " of iterator, axis " << bad_axis;
      ss << " of operand " << *bad_op_it;
      ss << " has an incorrect length " << op_sh.at(*bad_op_it).at(bad_axis);
      ss << " != " << size_dim_j;
      throw std::invalid_argument(ss.str());
    }

    shape.at(j) = size_dim_j;
  }
}

template<typename... Ts>
typename operands_container<Ts...>::iterator operands_container<Ts...>::begin() {
  return iterator(*this, std::vector<size_t>(ndim, 0lu), false);
}

template<typename... Ts>
typename operands_container<Ts...>::iterator operands_container<Ts...>::end() {
  return iterator(*this, std::vector<size_t>(ndim, 0lu), true);
}
