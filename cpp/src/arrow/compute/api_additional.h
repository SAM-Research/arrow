// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <memory>
#include <utility>

#include "api_aggregate.h"
#include "arrow/compute/function_options.h"
#include "arrow/compute/ordering.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace compute {

struct ARROW_EXPORT ShiftOptions : public FunctionOptions {
  explicit ShiftOptions(int32_t periods, std::shared_ptr<arrow::Scalar> fill_value)
      : FunctionOptions({}), periods(periods), fill_value(std::move(fill_value)) {}

  static constexpr char const kTypeName[] = "ShiftOptions";

  static ShiftOptions Defaults() { return ShiftOptions(1, nullptr); }

  int32_t periods;
  std::shared_ptr<arrow::Scalar> fill_value;
};

class ExecContext;

Result<Datum> Shift(const Datum& values, const ShiftOptions& options);

Result<arrow::Datum> Covariance(std::shared_ptr<arrow::Array> const& arrayx,
                                std::shared_ptr<arrow::Array> const& arrayy,
                                VarianceOptions const& options = VarianceOptions(1));

Result<arrow::Datum> Correlation(std::shared_ptr<arrow::Array> const& arrayx,
                                 std::shared_ptr<arrow::Array> const& arrayy,
                                 compute::VarianceOptions const& options);

arrow::Result<arrow::Datum> PctChange(const std::shared_ptr<arrow::Array>& input, int periods);

arrow::Result<arrow::Datum> AutoCorr(const std::shared_ptr<arrow::Array>& input, int lag = 1);
}  // namespace compute
}  // namespace arrow