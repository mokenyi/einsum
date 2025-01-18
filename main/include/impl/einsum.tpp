#include <tuple>
#include <string>
#include "xtensor/xexpression.hpp"
#include "implicit_out.hpp"
#include "has_repeated_labels.hpp"
#include "get_combined_dims_view.hpp"
#include "operands_container.hpp"
#include "einsum_util.hpp"

template<typename... Ss>
einsum<Ss...>::einsum(int e)
: ellipsis(e)
{
}

template<int I = 0, typename... Ss>
typename std::enable_if<I == sizeof...(Ss), void>::type get_label_counts_loop(
  std::tuple<Ss...> const& subs,
  std::array<size_t,sizeof...(Ss)> const& ndim,
  std::vector<std::array<int,2>>& label_counts,
  int ellipsis
) {
}

template<int I = 0, typename... Ss>
typename std::enable_if<I < sizeof...(Ss), void>::type get_label_counts_loop(
  std::tuple<Ss...> const& subs,
  std::array<size_t,sizeof...(Ss)> const& ndim,
  std::vector<std::array<int,2>>& label_counts,
  int ellipsis
) {
  // NB: We *don't* want combined labels here, otherwise a label of a pair of
  // combined dimensions would erroneously appear to be a free index.
  std::vector<int> const labels = std::tuple_element<I, std::tuple<Ss...>>::type::parse_operand_subscripts(ndim.at(I), ellipsis);

  for (int k=0; k<labels.size(); ++k) {
    int const i = labels.at(k) < 0 ? labels.at(k+labels.at(k)) : labels.at(k);

    if (i == ellipsis) {
      continue;
    }

    auto it = std::find_if(
      label_counts.begin(),
      label_counts.end(),
      [i](std::array<int,2> const& a) { return a.at(0) == i; }
    );

    if (it != label_counts.end()) {
      it->at(1)++;
    }
    else {
      label_counts.push_back(std::array<int,2>());
      label_counts.back().at(0) = i;
      label_counts.back().at(1) = 1;
    }
  }
  
  get_label_counts_loop<I+1>(subs, ndim, label_counts, ellipsis);
}

template<typename O>
typename std::enable_if<
  !std::is_same<O,implicit_out>::value,
  std::vector<int>
>::type get_output_labels(
  std::vector<std::array<int,2>> const& label_counts,
  int ndim_broadcast,
  int ellipsis
) {
  static_assert(
    !has_repeated_labels<O::start_first,O::start_second,typename O::tuple_type>::value,
    "output subscripts must not have repeated labels"
  );

  std::vector<int> labels = O::to_vector();
  int const length = labels.size();

  int const num_ellipses = std::count(labels.begin(), labels.end(), ellipsis);
  if (num_ellipses > 1) {
    std::stringstream ss;
    ss << "Found multiple ellipses in subscripts " << O::to_string();
    throw std::invalid_argument(ss.str());
  }

  auto ellipsis_it = std::find(labels.begin(), labels.end(), ellipsis);
  int const found_ellipsis = ellipsis_it == labels.end() ? -1 :
    std::distance(labels.begin(), ellipsis_it);

  if (found_ellipsis != -1) {
    for (int i=0; i<ndim_broadcast-1; ++i) {
      labels.insert(std::next(labels.begin(),found_ellipsis),ellipsis);
    }
  }
  else if (ndim_broadcast > 0) {
    throw std::invalid_argument("Output has more dimensions than subscripts "
      "given in Einstein sum, but no ellipsis provided to broadcast the "
      "extra dimensions");
  }

  for (int i=0; i<length; ++i) {
    int const label = labels.at(i);
    if (label != ellipsis) {
      auto label_count_it = std::find_if(
        label_counts.begin(),
        label_counts.end(),
        [label](std::array<int,2> const& l) { return l.at(0) == label; }
      );
      if (label_count_it == label_counts.end()) {
        std::stringstream ss;
        ss << "Output subscripts contain label " << label << " which does not ";
        ss << "appear in input subscripts";
        throw std::invalid_argument(ss.str());
      }
    }
  }

  return labels;
}

template<typename O>
typename std::enable_if<
  std::is_same<O,implicit_out>::value,
  std::vector<int>
>::type get_output_labels(
  std::vector<std::array<int,2>> const& label_counts,
  int ndim_broadcast,
  int ellipsis
) {
  std::vector<int> output_labels(ndim_broadcast, ellipsis);
  for (auto const& label_count: label_counts) {
    if (label_count.at(1) == 1) {
      output_labels.push_back(label_count.at(0));
    }
  }
  return output_labels;
}

template<size_t I = 0lu, typename U, typename... Ts>
typename std::enable_if<I == sizeof...(Ts),void>::type copy(
  std::array<U,sizeof...(Ts)>& dest,
  std::tuple<Ts...> const& src
) {
}

template<size_t I = 0lu, typename U, typename... Ts>
typename std::enable_if<I < sizeof...(Ts),void>::type copy(
  std::array<U,sizeof...(Ts)>& dest,
  std::tuple<Ts...> const& src
) {
  dest.at(I) = std::get<I>(src);
  copy<I+1>(dest, src);
}

template<size_t N>
std::array<std::vector<int>,N+1> prepare_op_axes_loop(
  std::array<std::vector<int>,N> const& combined_labels,
  std::vector<int> const& iter_labels,
  int ellipsis
) {
  std::array<std::vector<int>,N+1> op_axes;

  for (int j=0; j<N; ++j) {
    std::vector<int>& axes = op_axes.at(j);
    std::vector<int> const& labels = combined_labels.at(j);
    int const ndim_op = labels.size();
    int const ndim_iter = iter_labels.size();

    axes.resize(ndim_iter, -1);

    int ibroadcast = ndim_op-1;
    for (int i=ndim_iter-1; i>=0; --i) {
      int const label = iter_labels.at(i);
      if (label == ellipsis) {
        while (ibroadcast >= 0 && labels.at(ibroadcast) != 0) {
          --ibroadcast;
        }

        if (ibroadcast < 0) {
          axes.at(i) = -1;
        }
        else {
          axes.at(i) = ibroadcast;
          --ibroadcast;
        }
      }
      else {
        auto const match = std::find(labels.begin(), labels.end(), label);
        if (match == labels.end()) {
          axes.at(i) = -1;
        }
        else {
          axes.at(i) = std::distance(labels.begin(), match);
        }
      }
    }
  }

  return op_axes;
}

template<typename... Ss>
template<typename O, typename... Ts>
auto einsum<Ss...>::eval(xt::xexpression<Ts> const&... op_in) -> xt::xarray<typename std::common_type<typename Ts::value_type...>::type> {
  // TODO: Consider view_eval(op_in.derived_cast()) to allow op_in to work
  // with general xexpressions.
  constexpr size_t num_ops = sizeof...(op_in);

  // Pack expand the numbers of dimensions in the operands into array
  // for easy access.
  std::array<size_t,num_ops> ndim;
  get_ndim(std::tie(op_in.derived_cast()...), ndim);

  auto const ops = std::make_tuple(
    get_combined_dims_view<Ss>(
      op_in.derived_cast(),
      ellipsis
    )...
  );

  // [0] of each item is a label/subscript that occurs in Ss.
  // [1] of each item is the number of times that label occurs (i.e. output
  // labels will have [1] = 1).
  std::vector<std::array<int,2>> label_counts;

  // Pack expand the operand subscript types.
  auto const subs = std::make_tuple(Ss()...);

  get_label_counts_loop(subs, ndim, label_counts, ellipsis);
  //std::cout << "label_counts" << std::endl;
  //for (int i=0; i<label_counts.size(); ++i) {
  //  std::cout << label_counts.at(i).at(0) << " " << label_counts.at(i).at(1) << std::endl;
  //}
  
  std::array<std::vector<int>,num_ops> combined_labels;
  
  copy(
    combined_labels,
    std::make_tuple(Ss::get_combined_labels(op_in.derived_cast().dimension(),ellipsis)...)
  );

  //std::cout << "combined_labels" << std::endl;
  //for (int i=0; i<combined_labels.size(); ++i) {
  //  print_vector(combined_labels.at(i), std::cout);
  //  std::cout << std::endl;
  //}

  /*
   * In the subscripts of an operand, _ denotes broadcast
   * dimensions, which fall into two categories:
   * - unlabelled dimensions, which are implicitly free Einstein indexes
   *   so they appear in output. An unlabelled dimension in operand A
   *   may correspond to either
   *   - an unlabelled dim in op B if the position of the unlabelled dim
   *     in op A's set of broadcast dims is greater than or equal to
   *     (ndim(broadcast,A) - ndim(unlabelled, B)).
   *   - an inserted dim in op B otherwise.
   * - inserted dimensions, which are created by inserting an extra dim
   *   at the end of the operand's broadcast dimensions in its shape
   *   tuple.
   *
   * Find the number of broadcast dimensions, which is the maximum
   * number of unlabelled dimensions in an combined_labels array.
   *
   * The ops are broadcast such that the ellipses denote identical
   * slices of the shape tuples of the broadcasted ops e.g. consider
   * np.einsum('ij...,jk...',a,b) where a.shape == (3,3,5,3),
   * b.shape == (3,4,3). a has 2 unlabelled dims and 0 inserted dims.
   * b has 1 unlabelled dim and 1 inserted dim.
   * After broadcast (denoted with `)
   * a`.shape == (3,3,5,3), b`.shape == (3,4,5,3). The ellipsis denotes
   * a`.shape[2:] which also equals b`.shape[2:]. In general for n_op
   * ops, n(inserted, A) = (ndim_broadcast - n(unlabelled, A)) dims will
   * be added to op A during broadcast, where n(unlabelled, A) = number
   * of unlabelled dims in op A.
   *
   * To get the result for the above example, we iterate the broadcast
   * dimensions in each op simultaneously, doing the matrix product for
   * each value of l and m, say, which index the last two dimensions in
   * both ops.
   *
   * In the general case, the shape of the result will have dims
   * coming from the summation over non-broadcast dims in the
   * operands (i.e. those with explicit labels in the subscripts string) plus
   * ndim_broadcast broadcast dims.
   */
  int ndim_broadcast(-1);
  for (auto const& combined_label: combined_labels) {
    int const num_zeros = std::count(
      combined_label.begin(),
      combined_label.end(),
      ellipsis
    );

    ndim_broadcast = num_zeros > ndim_broadcast ? num_zeros : ndim_broadcast;
  }

  std::vector<int> const output_labels = get_output_labels<O>(
    label_counts,
    ndim_broadcast,
    ellipsis
  );

  //std::cout << "output_labels" << std::endl;
  //print_vector(output_labels, std::cout);
  //std::cout << std::endl;

  size_t const ndim_output = output_labels.size();

  // iter_labels comprise the labels in the output array (i.e. free indexes)
  // followed by the dummy indexes.
  std::vector<int> iter_labels(output_labels);
  for (auto const& label_count: label_counts){
    if (std::find(
          output_labels.begin(),
          output_labels.end(),
          label_count.at(0)) == output_labels.end()) {
      iter_labels.push_back(label_count.at(0));
    }
  }

  //std::cout << "iter_labels" << std::endl;
  //print_vector(iter_labels, std::cout);
  //std::cout << std::endl;

  int const ndim_iter = iter_labels.size();

  // dimension j in the iterator maps to dimension op_axes[i][j] in operand
  // i.
  std::array<std::vector<int>,num_ops+1> op_axes = prepare_op_axes_loop(
    combined_labels,
    iter_labels,
    ellipsis
  );

  std::vector<size_t> return_shape;

  std::vector<int>& output_op_axes = op_axes.at(num_ops);
  output_op_axes.resize(ndim_iter);
  std::iota(
    output_op_axes.begin(),
    std::next(output_op_axes.begin(),ndim_output),
    0
  );

  std::fill(
    std::next(output_op_axes.begin(),ndim_output),
    output_op_axes.end(),
    -1
  );


  //std::cout << "op_axes" << std::endl;
  //for (auto const& v: op_axes) {
  //  print_vector(v, std::cout);
  //  std::cout << std::endl;
  //}

  using output_value_type = typename std::common_type<typename Ts::value_type...>::type;

  std::array<size_t,num_ops> ndim_combined;
  get_ndim(ops, ndim_combined);

  std::array<std::vector<size_t>,num_ops> op_sh;
  set_op_sh(op_sh, ops);

  std::vector<size_t> const result_shape = get_output_shape(op_axes, op_sh);

  xt::xarray<output_value_type> result;
  result.resize(result_shape);
  result.fill(static_cast<output_value_type>(0));

  auto const operands_and_result = std::tuple_cat(ops, std::tie(result));

  auto einsum_ops = make_operands_container(operands_and_result, op_axes);
  
  for (auto c: einsum_ops) {
    *std::get<num_ops>(c) += product_of_pointees<0, num_ops>(c);
  }

  return result;
}
