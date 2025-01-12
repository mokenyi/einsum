#include "doctest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "einsum.hpp"
#include "subscripts.hpp"
#include "labels.hpp"

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

