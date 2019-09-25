struct GLMTrait{D<:ResponseDistribution, L<:InverseLinkFunction}
formula::String
mu::Vector{Float64}
dist:: D
link:: L
end

function GLMTrait(mu::Number, df, dist::D, link::L) where {D, L}
    return(GLMTrait{D, L}(string(mu), repeat([mu], size(df, 1)), dist, link))
end

function GLMTrait(formula::String, df, dist::D, link::L) where {D, L}
    mu = mean_formula(formula, df)
    return(GLMTrait{D, L}(formula, mu, dist, link))
end

function Multiple_GLMTraits(formulas, df, dist::ResponseDistribution, link::InverseLinkFunction)
  vec = [GLMTrait(formulas[i], df, dist, link) for i in 1:length(formulas)] #vector of GLMTrait objects
  return(vec)
end

# we put type of the dist vector as Any since we want to allow for any ResponseType{Poisson(), LogLink()}, ResponseType{Normal(), IdentityLink()}
function Multiple_GLMTraits(formulas::Vector{String}, df::DataFrame, dist::Vector, link::Vector)
  vec = [GLMTrait(formulas[i], df, dist[i], link[i]) for i in 1:length(formulas)]
  return(vec)
end


# lmm: multiple traits (MVN)

struct LMMTrait{T}
formulas::Vector{String}
mu::Matrix{Float64}
vc::T
  function LMMTrait(formulas, df, vc::T) where T
    n_traits = length(formulas)
    n_people = size(df)[1]
    mu = zeros(n_people, n_traits)
    for i in 1:n_traits
      #calculate the mean vector
      mu[:, i] += mean_formula(formulas[i], df)
    end
    return(new{T}(formulas, mu, vc))
  end
end