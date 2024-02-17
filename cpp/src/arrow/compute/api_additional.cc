//
// Created by dewe on 2/6/24.
//
#include "api_additional.h"
#include "api.h"


namespace arrow {
namespace compute {

Result<Datum> Shift(const Datum& values, const ShiftOptions& options) {
  return CallFunction("shift", {Datum(values)}, &options);
}

Result<arrow::Datum> Covariance(std::shared_ptr<arrow::Array> const& arrayx,
                                std::shared_ptr<arrow::Array> const& arrayy,
                                VarianceOptions const& options) {
  return CallFunction("cov", {arrayx, arrayy}, &options);
}

Result<arrow::Datum> Correlation(std::shared_ptr<arrow::Array> const& arrayx,
                                 std::shared_ptr<arrow::Array> const& arrayy,
                                 compute::VarianceOptions const& options) {
  return CallFunction("corr", {arrayx, arrayy}, &options);
}

arrow::Result<arrow::Datum> AutoCorr(const std::shared_ptr<arrow::Array>& input,
                                     int lag) {
  if (lag > input->length()) {
    return arrow::Status::Invalid("Lag cannot be greater than the length of the array");
  }
  // Shift the input array by the lag value
  ShiftOptions shiftOptions{lag, arrow::MakeNullScalar(input->type())};
  ARROW_ASSIGN_OR_RAISE(auto shifted_array,
                        CallFunction("shift", {input}, &shiftOptions));

  // Compute the covariance between the input array and the shifted array
  ARROW_ASSIGN_OR_RAISE(auto covariance, CallFunction("cov", {input, shifted_array}));
  // Compute the variance of the input array
  ARROW_ASSIGN_OR_RAISE(auto variance, Variance(input));
  // Divide the covariance by the variance to get the correlation
  ARROW_ASSIGN_OR_RAISE(auto correlation, CallFunction("divide", {covariance, variance}));
  // Divide the correlation by the number of observations minus the lag to get the autocorrelation
  return CallFunction("divide", {correlation, MakeScalar(input->length() - 1 - lag)});
}

arrow::Result<arrow::Datum> PctChange(const std::shared_ptr<arrow::Array>& input,
                                      int periods) {
  if (periods > input->length()) {
    return arrow::Status::Invalid(
        "Periods cannot be greater than the length of the array");
  }

  // Shift the input array by the number of periods
  ShiftOptions options{periods, arrow::MakeNullScalar(input->type())};
  ARROW_ASSIGN_OR_RAISE(auto shifted_array, CallFunction("shift", {input}, &options));

  // Divide the shifted array by the original array and subtract 1.0
  ARROW_ASSIGN_OR_RAISE(auto divided_array,
                        CallFunction("divide", {input, shifted_array}));
  return CallFunction("subtract", {divided_array, MakeScalar(1.0)});
}
}
}  // namespace arrow
