#ifndef MULTI_ITERATOR_HPP
#define MULTI_ITERATOR_HPP

// A version numpy's Array Iterator which simultaneously iterates over a
// set of xtensors.
// 
// T: type of data held by the arrays being iterated.
template<typename... Ts>
class multi_iterator {
public:
  template<typename... Ts>
  multi_iterator(Ts... x);
  size_t constexpr narg = sizeof...(Ts);

  using current_t = std::tuple<Ts::value_type*...>;
private:
  size_t ndim;
  std::tuple<Ts...> exp;
  std::tuple<Ts::iterator...> its;
  current_t current;
  std::array<size_t> shape;
  std::array<size_t> idx;
  // [i, j] = index in op i of the dim which dim j in iterator is mapped to.
  xt::xtensor<size_t, narg> op_axes; 
  void reset(size_t arg, size_t dim);
}

#include "impl/multi_iterator.tpp"

#endif // MULTI_ITERATOR_HPP
