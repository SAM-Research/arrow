#pragma once
#include <memory>
#include <string>
#include <utility>
#include "arrow/array.h"
#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/kernel.h"
#include "arrow/compute/kernels/aggregate_internal.h"
#include "arrow/compute/kernels/common_internal.h"
#include "arrow/status.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/int128_internal.h"


namespace arrow {
namespace compute {
namespace internal {

template <typename ValueType, typename SumType, SimdLevel::type SimdLevel,
          typename ValueFunc>
enable_if_t<std::is_floating_point<SumType>::value, SumType> SumArray2WithCovariance(
    const ArraySpan& data1, const ArraySpan& data2, ValueFunc&& func) {
  using arrow::internal::VisitSetBitRunsVoid;

  const int64_t data_size = data1.length - data1.GetNullCount();
  if (data_size == 0) {
    return 0;
  }

  // number of inputs to accumulate before merging with another block
  constexpr int kBlockSize = 16;  // same as numpy
  // levels (tree depth) = ceil(log2(len)) + 1, a bit larger than necessary
  const int levels = bit_util::Log2(static_cast<uint64_t>(data_size)) + 1;
  // temporary summation per level
  std::vector<SumType> sum(levels);
  // whether two summations are ready and should be reduced to upper level
  // one bit for each level, bit0 -> level0, ...
  uint64_t mask = 0;
  // level of root node holding the final summation
  int root_level = 0;

  // reduce summation of one block (may be smaller than kBlockSize) from leaf node continue reducing to upper level if two summations are ready for non-leaf node (capture `levels` by value because of ARROW-17567)
  auto reduce = [&, levels](SumType block_sum) {
    int cur_level = 0;
    uint64_t cur_level_mask = 1ULL;
    sum[cur_level] += block_sum;
    mask ^= cur_level_mask;
    while ((mask & cur_level_mask) == 0) {
      block_sum = sum[cur_level];
      sum[cur_level] = 0;
      ++cur_level;
      DCHECK_LT(cur_level, levels);
      cur_level_mask <<= 1;
      sum[cur_level] += block_sum;
      mask ^= cur_level_mask;
    }
    root_level = std::max(root_level, cur_level);
  };

  const auto* values1 = data1.GetValues<ValueType>(1);
  const auto* values2 = data2.GetValues<ValueType>(1);
  VisitSetBitRunsVoid(data1.buffers[0].data, data1.offset, data1.length,
                      [&](int64_t pos, int64_t len) {
                        const ValueType* v1 = &values1[pos];
                        const ValueType* v2 = &values2[pos];
                        // unsigned division by constant is cheaper than signed one
                        const uint64_t blocks = static_cast<uint64_t>(len) / kBlockSize;
                        const uint64_t remains = static_cast<uint64_t>(len) % kBlockSize;

                        for (uint64_t i = 0; i < blocks; ++i) {
                          SumType block_sum = 0;
                          for (int j = 0; j < kBlockSize; ++j) {
                            block_sum += func(v1[j], v2[j]);
                          }
                          reduce(block_sum);
                          v1 += kBlockSize;
                          v2 += kBlockSize;
                        }

                        if (remains > 0) {
                          SumType block_sum = 0;
                          for (uint64_t i = 0; i < remains; ++i) {
                            block_sum += func(v1[i], v2[i]);
                          }
                          reduce(block_sum);
                        }
                      });

  // reduce intermediate summations from all non-leaf nodes
  for (int i = 1; i <= root_level; ++i) {
    sum[i] += sum[i - 1];
  }

  return sum[root_level];
}

template <typename ArrowType>
struct IntegerCovariance {
  using c_type = typename ArrowType::c_type;

  int64_t count = 0;
  int64_t sum_x = 0;
  int64_t sum_y = 0;
  arrow::internal::int128_t sum_xy = 0;

  void ConsumeOne(const c_type value_x, const c_type value_y) {
    sum_x += value_x;
    sum_y += value_y;
    sum_xy += static_cast<arrow::internal::int128_t>(value_x) * value_y;
    count++;
  }

  double mean_x() const { return static_cast<double>(sum_x) / count; }

  double mean_y() const { return static_cast<double>(sum_y) / count; }

  double m_xy() const {
    // calculate m_xy = sum_xy - sum_x * sum_y / count
    // decompose sum_x * sum_y / count into integers and fractions
    const auto sum_square = static_cast<arrow::internal::int128_t>(sum_x) * sum_y;
    const arrow::internal::int128_t integers = sum_square / count;
    const double fractions = static_cast<double>(sum_square % count) / count;
    return static_cast<double>(sum_xy - integers) - fractions;
  }
};

template <typename ArrowType>
struct CovarianceState {
  using ArrayType = typename TypeTraits<ArrowType>::ArrayType;
  using CType = typename TypeTraits<ArrowType>::CType;
  using ThisType = CovarianceState<ArrowType>;

  CovarianceState(int32_t decimal_scale, VarianceOptions options)
      : count(0),
        mean_x(0),
        mean_y(0),
        m_xy(0),
        decimal_scale(decimal_scale),
        options(std::move(options)) {}

  template <typename T>
  double ToDouble(T value) const {
    return static_cast<double>(value);
  }
  double ToDouble(const Decimal128& value) const { return value.ToDouble(decimal_scale); }
  double ToDouble(const Decimal256& value) const { return value.ToDouble(decimal_scale); }

  template <typename T = ArrowType>
  enable_if_t<is_integer_type<T>::value && (sizeof(CType) <= 4)> Consume(
      const ArraySpan& array_x, const ArraySpan& array_y) {
    // max number of elements that sum will not overflow int64 (2Gi int32 elements) for uint32: 0 <= sum < 2^63 (int64 >= 0) for int32: -2^62 <= sum < 2^62
    constexpr int64_t max_length = 1ULL << (63 - sizeof(CType) * 8);
    this->all_valid = (array_x.GetNullCount() == 0) && (array_y.GetNullCount() == 0);
    if (!this->all_valid && !options.skip_nulls) return;
    int64_t start_index_x = 0;
    int64_t start_index_y = 0;
    int64_t valid_count_x = array_x.length - array_x.GetNullCount();
    int64_t valid_count_y = array_y.length - array_y.GetNullCount();
    ArraySpan slice_x = array_x;
    ArraySpan slice_y = array_y;
    while (valid_count_x > 0 && valid_count_y > 0) {
      // process in chunks that overflow will never happen
      slice_x.SetSlice(start_index_x + array_x.offset,
                       std::min(max_length, array_x.length - start_index_x));
      slice_y.SetSlice(start_index_y + array_y.offset,
                       std::min(max_length, array_y.length - start_index_y));
      const int64_t count_x = slice_x.length - slice_x.GetNullCount();
      const int64_t count_y = slice_y.length - slice_y.GetNullCount();
      start_index_x += slice_x.length;
      start_index_y += slice_y.length;
      valid_count_x -= count_x;
      valid_count_y -= count_y;

      if (count_x > 0 && count_y > 0) {
        IntegerCovariance<ArrowType> cov;
        const CType* values_x = slice_x.GetValues<CType>(1);
        const CType* values_y = slice_y.GetValues<CType>(1);
        VisitSetBitRunsVoid(slice_x.buffers[0].data, slice_x.offset, slice_x.length,
                            [&](int64_t pos, int64_t len) {
                              for (int64_t i = 0; i < len; ++i) {
                                const auto value_x = values_x[pos + i];
                                const auto value_y = values_y[pos + i];
                                cov.ConsumeOne(value_x, value_y);
                              }
                            });

        // merge variance
        ThisType state(decimal_scale, options);
        state.count = cov.count;
        state.mean_x = cov.mean_x();
        state.mean_y = cov.mean_y();
        state.m_xy = cov.m_xy();
        this->MergeFrom(state);
      }
    }
  }

  template <typename T = ArrowType>
  enable_if_t<is_floating_type<T>::value || (sizeof(CType) > 4)> Consume(
      const ArraySpan& array_x, const ArraySpan& array_y) {
    this->all_valid = (array_x.GetNullCount() == 0) && (array_y.GetNullCount() == 0);
    int64_t count_x = array_x.length - array_x.GetNullCount();
    int64_t count_y = array_y.length - array_y.GetNullCount();

    if (count_x != count_y) {
      throw std::runtime_error("valid values from array1 must equal array2");
    }
    int64_t _count = array_x.length - array_x.GetNullCount();
    if (_count == 0 || (!this->all_valid && !options.skip_nulls)) {
      return;
    }

    using SumType = typename internal::GetSumType<T>::SumType;
    auto sum_x = internal::SumArray<CType, SumType, SimdLevel::NONE>(array_x);
    auto sum_y = internal::SumArray<CType, SumType, SimdLevel::NONE>(array_y);

    const double _mean_x = ToDouble(sum_x) / _count;
    const double _mean_y = ToDouble(sum_y) / _count;

    double _m_xy = internal::SumArray2WithCovariance<CType, double, SimdLevel::NONE>(
        array_x, array_y, [this, _mean_x, _mean_y](CType value_x, CType value_y) {
          const double vX = ToDouble(value_x);
          const double vY = ToDouble(value_y);
          return (vX - _mean_x) * (vY - _mean_y);
        });

    this->count = _count;
    this->mean_x = _mean_x;
    this->mean_y = _mean_y;
    this->m_xy = _m_xy;
  }

  void MergeFrom(const CovarianceState& other) {
    this->all_valid = this->all_valid && other.all_valid;
    if (other.count == 0) {
      return;
    }

    if (this->count == 0) {
      this->count = other.count;
      this->mean_x = other.mean_x;
      this->mean_y = other.mean_y;
      this->m_xy = other.m_xy;
      return;
    }
    auto count1 = this->count;
    auto count2 = other.count;

    this->count = count1 + count2;

    auto MergeMean = [this, count1, count2](double mean1, double mean2) mutable {
      return (mean1 * count1 + mean2 * count2) / double(this->count);
    };

    double _mean_x = MergeMean(this->mean_x, other.mean_x);
    double _mean_y = MergeMean(this->mean_y, other.mean_y);

    auto m22 = other.m_xy;
    this->m_xy += m22 + count1 * (this->mean_x - _mean_x) * (this->mean_y - _mean_y) +
                  count2 * (other.mean_x - _mean_x) * (other.mean_y - _mean_y);

    this->mean_x = _mean_x;
    this->mean_y = _mean_y;
  }

  int64_t count;
  double mean_x;
  double mean_y;
  double m_xy;
  bool all_valid = true;
  const int32_t decimal_scale;
  const VarianceOptions options;
};

template<typename ArrowType>
struct CorrelationState
{
  double mx2;
  double my2;
  CovarianceState<ArrowType> covarianceState;

  CorrelationState(int32_t decimal_scale, const VarianceOptions& options)
      : mx2(0.0), my2(0.0), covarianceState(decimal_scale, options)
  {
  }

  using CType = typename TypeTraits<ArrowType>::CType;

  void Consume(const ArraySpan& x, const ArraySpan& y)
  {
    covarianceState.Consume(x, y);

    mx2 = internal::SumArray<CType, double, SimdLevel::NONE>(
        x,
        [this](CType value)
        {
          const double v = covarianceState.ToDouble(value);
          return (v - this->covarianceState.mean_x) * (v - this->covarianceState.mean_x);
        });

    my2 = internal::SumArray<CType, double, SimdLevel::NONE>(
        y,
        [this](CType value)
        {
          const double v = covarianceState.ToDouble(value);
          return (v - this->covarianceState.mean_y) * (v - this->covarianceState.mean_y);
        });
  }

  static inline void MergeVarStd2(
      int64_t count1,
      double mean1,
      int64_t count2,
      double mean2,
      double m22,
      double* out_m2)
  {
    double mean = (mean1 * count1 + mean2 * count2) / (count1 + count2);
    *out_m2 += m22 + count1 * (mean1 - mean) * (mean1 - mean) + count2 * (mean2 - mean) * (mean2 - mean);
  }

  void MergeFrom(const CorrelationState& other)
  {
    auto count1 = covarianceState.count;
    auto meanx = covarianceState.mean_x;
    auto mean_y = covarianceState.mean_y;
    covarianceState.MergeFrom(other.covarianceState);
    MergeVarStd2(count1, meanx, other.covarianceState.count, other.covarianceState.mean_x, other.mx2, &mx2);
    MergeVarStd2(count1, mean_y, other.covarianceState.count, other.covarianceState.mean_y, other.my2, &my2);
  }
};

}
}
}  // namespace arrow