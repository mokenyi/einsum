#ifndef HAS_REPEATED_LABELS_HPP
#define HAS_REPEATED_LABELS_HPP

#include <tuple>
#include "next_label_pair.hpp"

template<size_t I, size_t J, typename T>
struct has_repeated_labels {
  static constexpr bool value =
    std::is_same<
      typename std::tuple_element<I, T>::type,
      typename std::tuple_element<J, T>::type
    >::value || has_repeated_labels<
      next_label_pair<I,J>::first, next_label_pair<I,J>::second, T
    >::value;
};

template<typename T>
struct has_repeated_labels<0lu, 1lu, T> {
  static constexpr bool value = std::is_same<
      typename std::tuple_element<0lu, T>::type,
      typename std::tuple_element<1lu, T>::type
    >::value;
};

#endif // HAS_REPEATED_LABELS_HPP
