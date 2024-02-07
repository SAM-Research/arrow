#pragma once
#include "cov_state.h"


namespace arrow {
namespace compute {
namespace internal {


template<typename ArrowType>
struct CovarianceImpl : public ScalarAggregator
{
    using ThisType = CovarianceImpl<ArrowType>;
    using ArrayType = typename TypeTraits<ArrowType>::ArrayType;

    explicit CovarianceImpl(
        int32_t decimal_scale,
        const std::shared_ptr<DataType>& out_type,
        const VarianceOptions& options)
        : out_type(out_type), state(decimal_scale, options)
    {
    }

    Status Consume(KernelContext*, const ExecSpan& batch) override
    {
        if (batch[0].is_array() && batch[1].is_array())
        {
            this->state.Consume(batch[0].array, batch[1].array);
        }
        else
        {
            return Status::Invalid("Input must be arrays");
        }
        return Status::OK();
    }

    Status MergeFrom(KernelContext*, KernelState&& src) override
    {
        const auto& other = checked_cast<const ThisType&>(src);
        this->state.MergeFrom(other.state);
        return Status::OK();
    }

    Status Finalize(KernelContext*, Datum* out) override
    {
        if (state.count <= state.options.ddof || state.count < state.options.min_count ||
            (!state.all_valid && !state.options.skip_nulls))
        {
            out->value = std::make_shared<DoubleScalar>();
        }
        else
        {
            double covar = state.m_xy / double(state.count - state.options.ddof);
            out->value = std::make_shared<DoubleScalar>(covar);
        }
        return Status::OK();
    }

    std::shared_ptr<DataType> out_type;
    CovarianceState<ArrowType> state;
};

struct CovarianceInitState
{
    std::unique_ptr<KernelState> state;
    KernelContext* ctx;
    const DataType& in_type_x;
    const DataType& in_type_y;
    const std::shared_ptr<DataType>& out_type;
    const VarianceOptions& options;

    CovarianceInitState(
        KernelContext* ctx,
        const DataType& in_type_x,
        const DataType& in_type_y,
        const std::shared_ptr<DataType>& out_type,
        const VarianceOptions& options)
        : ctx(ctx), in_type_x(in_type_x), in_type_y(in_type_y), out_type(out_type), options(options)
    {
    }

    Status Visit(const DataType&)
    {
        return Status::NotImplemented("No covariance implemented");
    }

    Status Visit(const HalfFloatType&)
    {
        return Status::NotImplemented("No covariance implemented");
    }

    template<typename Type>
    enable_if_number<Type, Status> Visit(const Type&)
    {
        state.reset(new CovarianceImpl<Type>(/*decimal_scale=*/0, out_type, options));
        return Status::OK();
    }

    template<typename Type>
    enable_if_decimal<Type, Status> Visit(const Type&)
    {
        state.reset(new CovarianceImpl<Type>(checked_cast<const DecimalType&>(in_type_x).scale(), out_type, options));
        return Status::OK();
    }

    Result<std::unique_ptr<KernelState>> Create()
    {
        RETURN_NOT_OK(VisitTypeInline(in_type_x, this));
        RETURN_NOT_OK(VisitTypeInline(in_type_y, this));
        return std::move(state);
    }
};

inline Result<std::unique_ptr<KernelState>> CovarianceInit(KernelContext* ctx, const KernelInitArgs& args)
{
    CovarianceInitState visitor(
        ctx,
        *args.inputs[0].type,
        *args.inputs[1].type,
        args.kernel->signature->out_type().type(),
        static_cast<const VarianceOptions&>(*args.options));
    return visitor.Create();
}

static void AddCovarianceKernels(
    KernelInit init,
    const std::vector<std::shared_ptr<DataType>>& types,
    ScalarAggregateFunction* func)
{
    for (const auto& ty : types)
    {
        auto sig = KernelSignature::Make({ InputType(ty->id()), InputType(ty->id()) }, float64());
        AddAggKernel(std::move(sig), init, func);
    }
}

const FunctionDoc covariance_doc{ "Calculate the covariance of 2 numeric array",
                                  ("The covariance function computes the covariance of two arrays, "
                                   "array_x and array_y. The covariance measures the degree to"
                                   " which two random variables are linearly related. "
                                   "A positive covariance indicates that the variables "
                                   "increase together, while a negative covariance "
                                   "indicates that the variables vary in opposite directions."
                                   " A covariance of zero indicates that the variables are"
                                   " independent. The function supports both integer "
                                   "and floating-point arrays and skips null values "
                                   "if specified in the filters. The result is returned as a double."),
                                  { "array1", "array2" },
                                  "VarianceOptions" };

static std::shared_ptr<ScalarAggregateFunction> AddCovarianceKernels()
{
    static auto default_std_options = VarianceOptions::Defaults();
    auto func = std::make_shared<ScalarAggregateFunction>("cov", Arity::Binary(), covariance_doc, &default_std_options);
    AddCovarianceKernels(CovarianceInit, NumericTypes(), func.get());
    AddCovarianceKernels(CovarianceInit, { decimal128(1, 1), decimal256(1, 1) }, func.get());
    return func;
}

void RegisterScalarAggregateCovariance(FunctionRegistry* registry)
{
    DCHECK_OK(registry->AddFunction(AddCovarianceKernels()));
}

} // namespace internal

} // namespace compute
} // namespace arrow