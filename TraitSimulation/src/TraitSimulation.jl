module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it 
using StatsModels # useful distributions #lots more useful distributions
using LinearAlgebra
using Random

include("calculate_mean_vector.jl")

include("apply_inverse_link_new.jl")
export LogLink, IdentityLink, SqrtLink, ProbitLink, LogitLink, InverseLink, CauchitLink, CloglogLink

include("simulate_glm_trait_new.jl")
export PoissonResponse, NormalResponse, BinomialResponse, BernoulliResponse, GammaResponse, InverseGaussianResponse, TResponse, WeibullResponse #Exporting these from the Distributions package 


#this is the main functionality of this package, to run the actual simulation now for the split up responsedist type and linkfunction type
function GLM_trait_simulation(mu, dist::ResponseDistribution, link::InverseLinkFunction) 
  transmu = apply_inverse_link(mu, link)
  Simulated_Trait = simulate_glm_trait(transmu, dist)
  return(Simulated_Trait)
end

########

include("Multiple_traits.jl")

include("Model_Framework.jl")

"""
```
simulate(trait::GLMTrait)
```
this for GLM trait
"""
function simulate(trait::GLMTrait)
    simulated_trait = GLM_trait_simulation(trait.mu, trait.dist, trait.link)
    rep_simulation = DataFrame(trait1 = simulated_trait)
    return(rep_simulation)
end

function simulate(trait::GLMTrait, n_reps::Int64)
  rep_simulation = Vector{DataFrame}(undef, n_reps)
  for i in 1:n_reps
    rep_simulation[i] = simulate(trait) # store each data frame in the vector of dataframes rep_simulation
  end
    return(rep_simulation)
end

"""
```
simulate(trait::LMMTrait)
```
this for LMMtrait
"""
function simulate(trait::LMMTrait)
  rep_simulation = LMM_trait_simulation(trait.mu, trait.vc)
  return(rep_simulation)
end

function simulate(trait::LMMTrait, n_reps::Int64)
  rep_simulation = Vector{DataFrame}(undef, n_reps)
  for i in 1:n_reps
    rep_simulation[i] = simulate(trait)
  end
  return(rep_simulation)
end


export ResponseType, GLM_trait_simulation, mean_formula, VarianceComponent, append_terms!, LMM_trait_simulation
export GLMTrait, Multiple_GLMTraits, LMMTrait, simulate, @vc, vcobjtuple, SimulateMVN, SimulateMVN!, Aggregate_VarianceComponents!
export TResponse, WeibullResponse, PoissonResponse, NormalResponse, BernoulliResponse, BinomialResponse
export GammaResponse, InverseGaussianResponse, ExponentialResponse
export CauchitLink, CloglogLink, IdentityLink, InverseLink, LogitLink, LogLink, ProbitLink, SqrtLink
end #module

