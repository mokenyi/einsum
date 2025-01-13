#include <tuple>
#include <string>
#include "xtensor/xexpression.hpp"
#include "implicit_out.hpp"
#include "has_repeated_labels.hpp"
#include "get_combined_dims_view.hpp"
#include "operands_container.hpp"

#define MAXDIMS 10

template<typename... Ss>
einsum<Ss...>::einsum(int e)
: ellipsis(e)
{
}

template<int I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type get_ndim(
  std::tuple<Ts...> const& ops,
  std::array<size_t,sizeof...(Ts)>& ndim
) {
}

template<int I = 0, typename... Ts>
typename std::enable_if<I < sizeof...(Ts), void>::type get_ndim(
  std::tuple<Ts...> const& ops,
  std::array<size_t,sizeof...(Ts)>& ndim
) {
  ndim.at(I) = std::get<I>(ops).dimension();
  get_ndim<I+1>(ops, ndim);
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
  std::array<size_t,num_ops> ndim;
  get_ndim(std::make_tuple(op_in.derived_cast()...), ndim);

  auto ops = std::make_tuple(
    get_combined_dims_view<Ss>(
      op_in,
      ellipsis
    ).derived_cast()...
  );

  std::vector<std::array<int,2>> label_counts;

  auto const subs = std::make_tuple(Ss()...);

  get_label_counts_loop(subs, ndim, label_counts, ellipsis);
  
  std::array<std::vector<int>,num_ops> combined_labels;

  copy(
    combined_labels,
    std::make_tuple(Ss::get_combined_labels(op_in.derived_cast().dimension(),ellipsis)...)
  );

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

  size_t const ndim_output = output_labels.size();

  std::vector<int> iter_labels(output_labels);
  for (auto const& label_count: label_counts){
    iter_labels.push_back(label_count.at(0));
  }

  int const ndim_iter = iter_labels.size();

  std::array<std::vector<int>,num_ops+1> op_axes = prepare_op_axes_loop(
    combined_labels,
    iter_labels,
    ellipsis
  );

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

  using output_value_type = typename std::common_type<typename Ts::value_type...>::type;

  xt::xarray<output_value_type> result;

  auto operands_and_result = std::tuple_cat(ops, std::make_tuple(result));

  // auto iter = make_operandsmulti_iterator(operands_and_result, op_axes);
  // TODO: Do this for real!
 
  return result;
}
