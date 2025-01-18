#include "doctest.h"
#include "einsum.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "labels.hpp"
#include "subscripts.hpp"

TEST_CASE("Broadcast dimensions, explicit output labels") {

}

TEST_CASE("Broadcast dimensions, implicit output labels") {

}

TEST_CASE("No broadcast dimensions, implicit output labels") {
  xt::xtensor<double,2> const x = {
    {0.82450064, 0.58973638, 0.26651105, 0.33222087},
    {0.2477022 , 0.93260011, 0.29248315, 0.04318276}
  };
  xt::xtensor_fixed<double,xt::xshape<3,4>> const y = {
    {0.11357894, 0.66455996, 0.75776223, 0.74428668},
    {0.12294283, 0.26457651, 0.93717736, 0.68914559},
    {0.81280101, 0.43663051, 0.88750341, 0.32555251}
  };

  xt::xarray<double> const actual = einsum<subscripts<I,J>,subscripts<K,J>>(_)
    .eval<implicit_out>(x, y);

  xt::xarray<double> const expected = {
    { 0.9347806697505194, 0.7361135049751363, 1.2723366530831643},
    { 0.9016754821643249, 0.5810650867741545, 0.8821723088938472}
  };

  CHECK(actual == expected);
}

TEST_CASE("No broadcast dimensions, explicit output labels") {
  xt::xtensor<double,2> const x = {
    {0.82450064, 0.58973638, 0.26651105, 0.33222087},
    {0.2477022 , 0.93260011, 0.29248315, 0.04318276}
  };
  xt::xtensor_fixed<double,xt::xshape<3,4>> const y = {
    {0.11357894, 0.66455996, 0.75776223, 0.74428668},
    {0.12294283, 0.26457651, 0.93717736, 0.68914559},
    {0.81280101, 0.43663051, 0.88750341, 0.32555251}
  };

  xt::xarray<double> const actual = einsum<subscripts<I,J>,subscripts<K,J>>(_)
    .eval<subscripts<K,I>>(x, y);

  xt::xarray<double> const expected = {
    { 0.9347806697505194,  0.9016754821643249},
    { 0.7361135049751363,  0.5810650867741545},
    { 1.2723366530831643,  0.8821723088938472}
  };

  CHECK(actual == expected);
} 

