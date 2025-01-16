#ifndef OPERANDS_CONTAINER_HPP
#define OPERANDS_CONTAINER_HPP

#include "xtl/xtype_traits.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"
#include <iterator>

// A version numpy's Array Iterator which simultaneously iterates over a
// set of xtensors.
// 
// Ts: types of expression held by the arrays being iterated.
// TODO: Enforce constraint that each Ts is either an xtensor or xarray (i.e. container type?)
template<typename... Ts>
class operands_container {
private:
  static size_t constexpr num_ops = sizeof...(Ts);

public:
  using op_axes_type = std::array<std::vector<int>,num_ops>;
  operands_container(
    std::tuple<Ts...> const& x,
    op_axes_type const& op_axes
  );

  struct iterator {
    using current_t = std::tuple<typename Ts::value_type*...>;

    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = current_t;
    using pointer           = current_t*;  // or also value_type*
    using reference         = current_t&;  // or also value_type&
    using const_reference   = current_t const&;
    friend class operands_container<Ts...>;
    const_reference operator*() const { return current; }
    pointer operator->() { return &current; }

    // Prefix increment
    iterator& operator++();

    // Postfix increment
    iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }
    bool operator==(iterator const& other);
    bool operator!=(iterator const& other);

    void print_current();

  private:
    iterator(
      operands_container<Ts...>& parent,
      std::vector<size_t>&& idx,
      bool is_end
    );
    std::vector<size_t> idx;
    current_t current;
    // Needs to be non-const so that I can mutate the values of what current
    // points to.
    operands_container<Ts...>& parent;
    bool is_end;
  };

  iterator begin();
  iterator end();
private:
  using size_type = size_t;
  size_t ndim;
  std::tuple<Ts...> exp;
  std::vector<size_t> shape;
  // [i, j] = index in op i of the dim which dim j in iterator is mapped to.
  op_axes_type op_axes;
  std::array<size_t, num_ops> op_idx;
  std::array<std::vector<size_t>,num_ops> op_sh;
  void reset(size_t arg, size_t dim);
  void set_shape();
};

#include "impl/operands_container.tpp"
#include "impl/operands_container_iterator.tpp"

template<typename... Ts>
operands_container<Ts...> make_operands_container(
  std::tuple<Ts...> const& t,
  typename operands_container<Ts...>::op_axes_type const& op_axes
) {
  // TODO: Assert that they are all xcontainers.
  //static_assert(
  //  xtl::conjunction<
  //    typename std::integral_constant<bool, xt::is_xexpression(Ts)::value>...
  //  >::value,
  //  "All parameters to operands_container must be xexpressions"
  //);

  return operands_container<Ts...>(t, op_axes);
}
#endif // OPERANDS_CONTAINER_HPP
