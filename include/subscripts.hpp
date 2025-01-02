#ifndef SUBSCRIPTS_HPP
#define SUBSCRIPTS_HPP
#include <cstdlib>
#include <tuple>
#include <vector>

template<size_t I = 0lu, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type push_value(std::vector<int>& dest, std::tuple<Ts...> const& src) {
}

template<size_t I = 0lu, typename... Ts>
typename std::enable_if<I < sizeof...(Ts), void>::type push_value(std::vector<int>& dest, std::tuple<Ts...> const& src) {
  dest.push_back(std::tuple_element<I, std::tuple<Ts...>>::type::value);
  push_value<I+1>(dest, src);
}

template<int... I>
struct subscripts {
  static constexpr size_t start_first = sizeof...(I) - 2;
  static constexpr size_t start_second = sizeof...(I) - 1;
  using tuple_type = std::tuple<std::integral_constant<int,I>...>;

  static std::vector<int> to_vector() {
    std::vector<int> ret;
    push_value(ret, tuple_type());
    return ret;
  }
};

#endif // SUBSCRIPTS_HPP
