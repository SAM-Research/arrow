#pragma once
#include "arrow/compute/api_additional.h"


namespace arrow {
namespace compute {
namespace internal {


class ShiftImpl
{

public:
    int32_t shift_value_{};
    std::shared_ptr<arrow::Scalar> replace_value_;

    ShiftImpl(int32_t shift_value, const std::shared_ptr<arrow::Scalar>& replace_value)
        : shift_value_(shift_value), replace_value_(replace_value)
    {
    }

    arrow::Status Execute(const std::shared_ptr<arrow::Array>& x, ExecResult* out)
    {
        auto builder = arrow::MakeBuilder(x->type()).MoveValueUnsafe();

        int64_t N = x->length();
        RETURN_NOT_OK(builder->Reserve(N));
        int64_t shift_len = std::abs(shift_value_);

        if (shift_value_ > 0)
        {
            if (replace_value_)
            {
                RETURN_NOT_OK(builder->AppendScalar(*replace_value_, shift_len));
            }
            else
            {
                RETURN_NOT_OK(builder->AppendNulls(shift_len));
            }

            for (int i = 0; i < N - shift_len; i++)
            {
                RETURN_NOT_OK(builder->AppendScalar(*x->GetScalar(i).MoveValueUnsafe()));
            }
        }
        else
        {
            for (int64_t i = shift_len; i < N; i++)
            {
                RETURN_NOT_OK(builder->AppendScalar(*x->GetScalar(i).MoveValueUnsafe()));
            }

            arrow::Status s;
            if (replace_value_)
            {
                RETURN_NOT_OK(builder->AppendScalar(*replace_value_, shift_len));
            }
            else
            {
                RETURN_NOT_OK(builder->AppendNulls(shift_len));
            }
        }
        std::shared_ptr<ArrayData> out_array;
        ARROW_RETURN_NOT_OK(builder->FinishInternal(&out_array));
        out->value = out_array;
        return arrow::Status::OK();
    }
};


struct ShiftKernel
{
    static arrow::Status Exec(KernelContext* ctx, const ExecSpan& batch, ExecResult* out)
    {
        const auto& options = OptionsWrapper<ShiftOptions>::Get(ctx);
        return ShiftImpl{ options.periods, options.fill_value }.Execute(batch.values.at(0).array.ToArray(), out);
    }
};

const FunctionDoc shift_doc{ "Shift the values of an input array by a given number of periods",
                             ("values must be numeric or boolean. The output is an array/chunked"
                              " array where each element has been shifted by the specified number "
                              "of periods. If the number of periods is negative, the array is shifted "
                              "to the left and the new values will be replaced with the specified fill"
                              " value or null if none is provided. If the number of periods is positive,"
                              " the array is shifted to the right and the new values will be replaced "
                              "with the specified fill value or null if none is provided."),
                             { "values" },
                             "ShiftOptions" };

void RegisterShiftFunction(FunctionRegistry* registry)
{
    static const ShiftOptions kDefaultOptions = ShiftOptions::Defaults();
    auto func = std::make_shared<VectorFunction>("shift", Arity::Unary(), shift_doc, &kDefaultOptions);

    std::vector<std::shared_ptr<DataType>> types;
    types.insert(types.end(), NumericTypes().begin(), NumericTypes().end());
    types.insert(types.end(), TemporalTypes().begin(), TemporalTypes().end());
    types.insert(types.end(), BinaryTypes().begin(), BinaryTypes().end());
    types.insert(types.end(), StringTypes().begin(), StringTypes().end());

    for (const auto& ty : types)
    {
        VectorKernel kernel;
        kernel.can_execute_chunkwise = false;
        kernel.null_handling = NullHandling::type::COMPUTED_NO_PREALLOCATE;
        kernel.mem_allocation = MemAllocation::type::NO_PREALLOCATE;
        kernel.signature = KernelSignature::Make({ ty }, OutputType(ty));
        kernel.exec = ShiftKernel::Exec;
        kernel.init = OptionsWrapper<ShiftOptions>::Init;
        DCHECK_OK(func->AddKernel(std::move(kernel)));
    }

    DCHECK_OK(registry->AddFunction(std::move(func)));
}

} // namespace internal
} // namespace compute
} // namespace arrow