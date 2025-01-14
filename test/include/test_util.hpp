#ifndef TEST_UTIL_HPP
#define TEST_UTIL_HPP

template<int I=0,typename... Ts>
typename std::enable_if<I==sizeof...(Ts),bool>::type equal_to_pointees(
  std::tuple<Ts...> const& a,
  std::tuple<Ts*...> const& b
) {
  return true;
}

template<int I=0,typename... Ts>
typename std::enable_if<I < sizeof...(Ts),bool>::type equal_to_pointees(
  std::tuple<Ts...> const& a,
  std::tuple<Ts*...> const& b
) {
  return std::get<I>(a) == *std::get<I>(b) && equal_to_pointees<I+1>(a,b);
}

template<typename... Ts>
bool equal_to_pointees(
  std::tuple<Ts*...> const& a,
  std::tuple<Ts...> const& b
) {
  return equal_to_pointees(b,a);
}
#endif // TEST_UTIL_HPP
