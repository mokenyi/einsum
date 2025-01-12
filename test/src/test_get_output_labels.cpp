#include "doctest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "einsum.hpp"
#include "subscripts.hpp"
#include "labels.hpp"

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

