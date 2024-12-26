#ifndef MULTI_ITERATOR_HPP
#define MULTI_ITERATOR_HPP

// A version numpy's Array Iterator which simultaneously iterates over a
// set of xtensors.
// 
// Ts: types of expression held by the arrays being iterated.
// TODO: Enforce constraint that each Ts is either an xtensor or xarray (i.e. container type?)
template<typename... Ts>
class multi_iterator {
  // TODO: Assert that they are all xcontainers.
  static_assert(
    xtl::conjunction<xt::is_xexpression(Ts)...>,
    "All parameters to multi_iterator must be xexpressions"
  );


public:
  template<typename... Ts>
  multi_iterator(Ts... x);
  size_t constexpr num_ops = sizeof...(Ts);

  using current_t = std::tuple<Ts::value_type const*...>;
private:
  using size_type = std::common_type<Ts::size_type...>;
  size_t ndim;
  std::tuple<Ts...> exp;
  current_t current;
  std::vector<size_t> shape;
  std::vector<size_t> idx;
  // [i, j] = index in op i of the dim which dim j in iterator is mapped to.
  xt::xtensor<int,2> op_axes; 
  std::array<size_type, num_ops> op_idx;
  xt::xtensor<size_t,2> op_sh;
  void reset(size_t arg, size_t dim);
  void set_shape();
  
  template<typename U, typename... Us>
  size_t get_max_op_ndim(U firstOp, Us... otherOps);
  size_t get_max_op_ndim();

  template<typename U, typename... Us>
  void set_op_sh(U firstOp, Us... otherOps);
  void set_op_sh();
}

#include "impl/multi_iterator.tpp"

#endif // MULTI_ITERATOR_HPP
