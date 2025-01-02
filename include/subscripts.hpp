#ifndef SUBSCRIPTS_HPP
#define SUBSCRIPTS_HPP
#include <cstdlib>
#include <tuple>

template<size_t... I>
struct subscripts {
  static constexpr size_t start_first = sizeof...(I) - 2;
  static constexpr size_t start_second = sizeof...(I) - 1;
  using tuple_type = std::tuple<std::integral_constant<size_t,I>...>;
};

#endif // SUBSCRIPTS_HPP
