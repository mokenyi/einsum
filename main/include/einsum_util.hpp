#ifndef EINSUM_UTIL_HPP
#define EINSUM_UTIL_HPP

#include <array>
#include <tuple>
#include <numeric>
#include <sstream>
#include <vector>
#include <algorithm>

template<size_t I=0, size_t N, typename... Ts>
typename std::enable_if<I == N, typename std::common_type<Ts...>::type>::type
product_of_pointees(
  std::tuple<Ts*...> const& pointers
) {
  return static_cast<typename std::common_type<Ts...>::type>(1);
}

template<size_t I=0, size_t N, typename... Ts>
typename std::enable_if<I < N, typename std::common_type<Ts...>::type>::type
product_of_pointees(
  std::tuple<Ts*...> const& pointers
) {
  return *std::get<I>(pointers) * product_of_pointees<I+1,N>(pointers);
}

template<int I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type get_ndim(
  std::tuple<Ts...> const& ops,
  std::array<size_t,sizeof...(Ts)>& ndim
) {
}

template<int I = 0, typename... Ts>
typename std::enable_if<I < sizeof...(Ts), void>::type get_ndim(
  std::tuple<Ts...> const& ops,
  std::array<size_t,sizeof...(Ts)>& ndim
) {
  ndim.at(I) = std::get<I>(ops).dimension();
  get_ndim<I+1>(ops, ndim);
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N == sizeof...(Ts), void>::type set_op_sh(
  std::array<std::vector<size_t>,sizeof...(Ts)>& op_sh,
  std::tuple<Ts...> const& x
) {
}

template<size_t N = 0LU, typename... Ts>
typename std::enable_if<N < sizeof...(Ts), void>::type set_op_sh(
  std::array<std::vector<size_t>,sizeof...(Ts)>& op_sh,
  std::tuple<Ts...> const& x
) {
  auto const& op = std::get<N>(x);
  op_sh.at(N).assign(op.shape().begin(), op.shape().end());;
  set_op_sh<N+1>(op_sh, x);
}

template<size_t N>
std::vector<size_t> get_output_shape(
  std::array<std::vector<int>,N+1> const& op_axes,
  std::array<std::vector<size_t>,N> const& op_sh
) {
  std::vector<int> const& output_op_axes = op_axes.back();
  auto const ndim_output_it =
    std::find(output_op_axes.begin(), output_op_axes.end(), -1);
  size_t const ndim_output = std::distance(
    output_op_axes.begin(),
    ndim_output_it
  );

  std::array<int,N> seq;
  std::iota(seq.begin(), seq.end(), 0);

  std::vector<size_t> output_shape(ndim_output,0lu);
  for (size_t j=0; j<ndim_output; ++j) {
    auto const mapped_op = std::find_if(
        seq.begin(),
        seq.end(),
        [&op_axes,j](int k) {
          return op_axes.at(k).at(j) != -1;
        }
    );
    if (mapped_op == seq.end()) {
      std::stringstream ss("Failed to get size of output dimension ");
      ss << j;
      throw std::invalid_argument(ss.str());
    }

    output_shape.at(j) = op_sh
      .at(*mapped_op)
      .at(op_axes.at(*mapped_op).at(j));
  }

  return output_shape;
}

#endif // EINSUM_UTIL_HPP
