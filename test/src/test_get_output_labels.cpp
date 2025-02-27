#include "doctest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "einsum.hpp"
#include "subscripts.hpp"
#include "labels.hpp"

TEST_CASE("Returns implicitly deduced output labels") {
  std::vector<std::array<int,2>> label_counts;
  label_counts.push_back(std::array<int,2>({A,1}));
  label_counts.push_back(std::array<int,2>({B,1}));
  label_counts.push_back(std::array<int,2>({C,1}));
  label_counts.push_back(std::array<int,2>({D,2}));

  int const ndim_broadcast(4);
  int const ellipsis(_);

  std::vector<int> const actual = get_output_labels<implicit_out>(
    label_counts,
    ndim_broadcast,
    ellipsis
  );

  std::vector<int> expected(ndim_broadcast, _);
  expected.push_back(A);
  expected.push_back(B);
  expected.push_back(C);

  CHECK(actual == expected);
}

TEST_CASE("Returns output labels when broadcast dims are places in the middle "
  "of the output array's shape") {
  using out_labels = subscripts<A,B,_,C>;
  std::vector<std::array<int,2>> label_counts;
  label_counts.push_back(std::array<int,2>({A,1}));
  label_counts.push_back(std::array<int,2>({B,1}));
  label_counts.push_back(std::array<int,2>({C,1}));
  label_counts.push_back(std::array<int,2>({D,2}));

  int const ndim_broadcast(4);
  int const ellipsis(_);

  std::vector<int> const actual = get_output_labels<out_labels>(
    label_counts,
    ndim_broadcast,
    ellipsis
  );

  std::vector<int> expected;
  expected.push_back(A);
  expected.push_back(B);
  for (int i=0; i<ndim_broadcast; ++i) { expected.push_back(_); }
  expected.push_back(C);

  CHECK(actual == expected);
}

TEST_CASE("Throw an exception if an output label does not appear in the input "
  "labels") {
  using out_labels = subscripts<A,B,C>;
  std::vector<std::array<int,2>> label_counts;
  label_counts.push_back(std::array<int,2>({A,1}));
  label_counts.push_back(std::array<int,2>({B,1}));

  CHECK_THROWS_WITH_AS(
    get_output_labels<out_labels>(label_counts,0,_),
    "Output subscripts contain label 3 which does not appear in input subscripts",
    std::invalid_argument
  );
}

// Uncommenting this case causes compilation to fail.
//TEST_CASE("get_output_labels should violate an assertion if the labels have "
//  "repeated values") {
//  using s1 = subscripts<A,B,A>;
//  std::vector<std::array<int,2>> label_counts;
//  label_counts.push_back(std::array<int,2>({A,2}));
//  label_counts.push_back(std::array<int,2>({B,1}));
//
//  get_output_labels<s1>(label_counts,0,_);
//}

