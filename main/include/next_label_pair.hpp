#ifndef NEXT_LABEL_PAIR_HPP
#define NEXT_LABEL_PAIR_HPP

#include <cstdlib>

template<size_t I, size_t J>
struct next_label_pair {
  static constexpr size_t first  = I == 0 ? J-2 : I-1;
  static constexpr size_t second = I == 0 ? J-1 : J;
};

#endif // NEXT_LABEL_PAIR_HPP

