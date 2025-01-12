#include "doctest.h"
#include "einsum.hpp"
#include "subscripts.hpp"

enum Labels: int {
  _,
  A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
};

TEST_CASE("get_label_counts should count occurrences of labels of uncombined "
  "dimensions in a set of subscripts") {
  using s1 = subscripts<I,I,_,J,K>;
  using s2 = subscripts<K,L,M,M>;
  using s3 = subscripts<N,O>;

  std::array<size_t,3> ndim = {6,4,2};
  std::vector<std::array<int,2>> actual;

  auto subs = std::make_tuple(s1(),s2(),s3());
  get_label_counts_loop(subs, ndim, actual, _);

  std::vector<std::array<int,2>> expected(7);
  expected.at(0) = {9,2};
  expected.at(1) = {10,1};
  expected.at(2) = {11,2};
  expected.at(3) = {12,1};
  expected.at(4) = {13,2};
  expected.at(5) = {14,1};
  expected.at(6) = {15,1};

  CHECK(actual == expected);
}

