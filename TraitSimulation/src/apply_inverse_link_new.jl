#for every different type of linkfunction, internally julia will find the right apply_inverse_link function

# This super type of all response distribution types
abstract type InverseLinkFunction end

##LINK FUNCTIONS##

"""inverse cauchit link."""

struct CauchitLink <: InverseLinkFunction
end

function cauchit_inverse_link(x)
  return atan(x) / pi + one(x) / 2
end

"""inverse cloglog link."""

struct CloglogLink <: InverseLinkFunction
end

function cloglog_inverse_link(x)
  return one(x) - exp(-exp(x))
end 

"""inverse identity link."""

struct IdentityLink <: InverseLinkFunction
end

function identity_inverse_link(x)
  return x
end

"""inverse inverse link."""

struct InverseLink <: InverseLinkFunction
end

function inverse_inverse_link(x)
  return one(x) / x
end

"""inverse logit link."""

struct LogitLink <: InverseLinkFunction
end

function logit_inverse_link(x)
  return one(x) / (one(x) + exp(-x))
end

"""inverse log link."""

struct LogLink <: InverseLinkFunction
end

function log_inverse_link(x)
  return exp(x)
end

"""inverse probit link."""
 
struct ProbitLink <: InverseLinkFunction
end


function probit_inverse_link(x)
  return (one(x) + erf(x / sqrt(2 * one(x)))) / 2
end

"""inverse sqrt link."""

struct SqrtLink <: InverseLinkFunction
end

function sqrt_inverse_link(x)
  return x * x
end

##APPLY INVERSE LINK FUNCTIONS

apply_inverse_link(μ, link::LogLink) = log_inverse_link.(μ)

apply_inverse_link(μ, link::IdentityLink) = identity_inverse_link.(μ)

apply_inverse_link(μ, link::SqrtLink) = sqrt_inverse_link.(μ)

apply_inverse_link(μ, link::ProbitLink) = probit_inverse_link.(μ)

apply_inverse_link(μ, link::LogitLink) = logit_inverse_link.(μ)

apply_inverse_link(μ, link::InverseLink) = inverse_inverse_link.(μ)

apply_inverse_link(μ, link::CauchitLink) = cauchit_inverse_link.(μ)

apply_inverse_link(μ, link::CloglogLink) = cloglog_inverse_link.(μ)

