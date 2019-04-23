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
function actual_simulation(mu, dist::ResponseDistribution, link::InverseLinkFunction) 
  transmu = apply_inverse_link(mu, link)
  Simulated_Trait = simulate_glm_trait(transmu, dist)
  return(Simulated_Trait)
end

########

include("Multiple_traits.jl")

include("Model_Framework.jl")

#this for GLM trait 
function simulate(trait::GLMTrait)
  simulated_trait = actual_simulation(trait.mu, trait.dist, trait.link)
  out = DataFrame(trait1 = simulated_trait)
  return(out)
end

# for multiple GLM traits 
function simulate(traits::Vector)
  simulated_traits = [actual_simulation(traits[i].mu, traits[i].dist, traits[i].link) for i in 1:length(traits)]
  out = DataFrame(simulated_traits)
  out = names!(out, [Symbol("trait$i") for i in 1:length(traits)])
  return(out)
end

# for LMMtrait
function simulate(trait::LMMTrait)
  LMM_trait_simulation(trait.mu, trait.vc)
end

export ResponseType, actual_simulation, mean_formula, VarianceComponent, append_terms!
export GLMTrait, Multiple_GLMTraits, LMMTrait, simulate, @vc, vcobjtuple
export TResponse, WeibullResponse, PoissonResponse, NormalResponse, BernoulliResponse, BinomialResponse
export GammaResponse, InverseGaussianResponse, ExponentialResponse
export CauchitLink, CloglogLink, IdentityLink, InverseLink, LogitLink, LogLink, ProbitLink, SqrtLink
end #module

