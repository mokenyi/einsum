#ifndef MULTI_ITERATOR_HPP
#define MULTI_ITERATOR_HPP

#include "xtl/xtype_traits.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"

// A version numpy's Array Iterator which simultaneously iterates over a
// set of xtensors.
// 
// Ts: types of expression held by the arrays being iterated.
// TODO: Enforce constraint that each Ts is either an xtensor or xarray (i.e. container type?)
template<typename... Ts>
class multi_iterator {
private:
  static size_t constexpr num_ops = sizeof...(Ts);

public:
  using op_axes_type = std::array<std::vector<int>,num_ops>;
  multi_iterator(
    std::tuple<Ts...> const& x,
    op_axes_type const& op_axes
  );

  void next();
  bool hasnext();
  void print_current();
  using current_t = std::tuple<typename Ts::value_type*...>;
private:
  using size_type = size_t;
  size_t ndim;
  std::tuple<Ts...> exp;
  current_t current;
  std::vector<size_t> shape;
  std::vector<size_t> idx;
  // [i, j] = index in op i of the dim which dim j in iterator is mapped to.
  op_axes_type op_axes;
  std::array<size_t, num_ops> op_idx;
  xt::xtensor<int,2> op_sh;
  void reset(size_t arg, size_t dim);
  void set_shape();
};

#include "impl/multi_iterator.tpp"

template<typename... Ts>
multi_iterator<Ts...> make_multi_iterator(
  std::tuple<Ts...> const& t,
  typename multi_iterator<Ts...>::op_axes_type const& op_axes
) {
  // TODO: Assert that they are all xcontainers.
  //static_assert(
  //  xtl::conjunction<
  //    typename std::integral_constant<bool, xt::is_xexpression(Ts)::value>...
  //  >::value,
  //  "All parameters to multi_iterator must be xexpressions"
  //);

  return multi_iterator<Ts...>(t, op_axes);
}
#endif // MULTI_ITERATOR_HPP
