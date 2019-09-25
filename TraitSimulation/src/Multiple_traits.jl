using DataFrames
using LinearAlgebra
using TraitSimulation
using Random

#this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop

struct VarianceComponent
	A::Matrix{Float64} # n_traits by n_traits
	B::Matrix{Float64} # n_people by n_people
	CholA::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of A
	CholB::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of B

	function VarianceComponent(A, B) #inner constructor given A, B 
		return(new(A, B, cholesky(A), cholesky(B))) # stores these values (this is helpful so we don't have it inside the loop)
	end
end

#single LMM trait with given evaluated matrix of Variance components 
function LMM_trait_simulation(mu, vc::Matrix{T}) where T
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	#preallocate memory for the returned dataframe simulated_trait
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)

	#for a single evaluated matrix as the specified covariance matrix instead of a Variancewe do not need to call the 
	chol_Σ = cholesky(vc)
	#generate from standard normal
	randn!(z)

	# we want to solve u then v to get the first variane component, v.
	#first matrix vector multiplication using cholesky decomposition

	#need to find which will be CholA, CholB 
	lmul!(chol_Σ.L, z)

	simulated_trait += z

	#for each trait
	simulated_trait += mu

	out = DataFrame(simulated_trait)

	out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

	return out
end



function append_terms!(AB, summand)
	# elements in args are symbols,
	A_esc = esc(summand.args[2])
	B_esc = esc(summand.args[3])
	push!(AB.args, :(VarianceComponent($A_esc, $B_esc)))
end

"""
this vc macro allows us to create a vector of VarianceComponent objects for simulation
"""

macro vc(expression)
	n = length(expression.args)
	# AB is an empty vector of variance components list of symbols
	AB = :(VarianceComponent[]) 
	if expression.args[1] != :+ #if first argument is not plus (only one vc)
		summand = expression 
		append_terms!(AB, summand)
	else #MULTIPLE VARIANCE COMPONENTS if the first argument is a plus (Sigma is a sum multiple variance components)
		for i in 2:n
			summand = expression.args[i]
			append_terms!(AB, summand)
		end
	end
	return(:($AB)) # change this to return a vector of VarianceComponent objects
end 


"""
this is a test for vcobjtuple that is compatible with VarianceComponentModels.jl
"""
function  vcobjtuple(vcobject::Vector{VarianceComponent})
	m = length(vcobject)
	d = size(vcobject[1].A, 1)
	n = size(vcobject[1].B, 1)

	Σ = ntuple(x -> zeros(d, d), m)
	V = ntuple(x -> zeros(n, n), m)

	for i in eachindex(vcobject)
		copyto!(V[i], vcobject[i].B)
		copyto!(Σ[i], vcobject[i].A)
	end
	return(Σ, V)
end


# for a single Variance Component
# algorithm that will transform z ~ N(0,1)
function SimulateMVN!(z, vc::VarianceComponent)
	#for the ith variance component (VC)
	cholA = vc.CholA # grab (not calculate) the stored Cholesky decomposition of n_traits by n_traits variance component matrix
	cholB = vc.CholB # grab (not calculate) the stored Cholesky decomposition of n_people by n_people variance component matrix

	#Generating MN(0, Sigma)
	# first generate from standard normal
	randn!(z)

	# we want to solve u then v to get the first variance component, v.
	# first matrix vector multiplication using the cholesky decomposed CholA, CholB above 
	lmul!(cholB.U, z)
	rmul!(z, cholA.L)

	#second matrix vector multiplication using the he cholesky decomposed CholA, CholB above
	rmul!(z, cholA.U)
	lmul!(cholB.L, z) #multiply on left and save to simulated_trait

	#add the effects of each variance component
	return(z)
end

#this function will call the SimulateMVN! so that I write over z (reuse memory allocation) for potentially many simulations
function SimulateMVN(n_people, n_traits, vc::VarianceComponent)
	#preallocate memory for the returned dataframe simulated_trait once
	z = Matrix{Float64}(undef, n_people, n_traits)

	# calls the function to apply the cholesky decomposition (we have stored in vc object)
	# and transforms/updates z ~ MN(0, vc)
	SimulateMVN!(z, vc::VarianceComponent)
	#returns the allocated and now transformed z
	return(z)
end

##Single Variance Component object 
#without computing mean from dataframe and formulas i.e given an evaluated matrix of means
function LMM_trait_simulation(mu, vc::VarianceComponent)
	n_people = size(mu)[1]
	n_traits = size(mu)[2]

	z = SimulateMVN(n_people, n_traits, vc)

	#for each trait add the mean --> MN(mu, Sigma)
	z += mu

	out = DataFrame(z)

	out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

	return out
end

# we note the exclamation is to indicate this function will mutate or override the values that its given
# here z, simulated_trait will get updated but vc will not be touched.
## AGGREGATE MULTIPLE VARIANCE COMPONENTS IN LMM TRAIT OBJECT to creat overall variance 
function Aggregate_VarianceComponents!(z, total_variance, vc::Vector{VarianceComponent})
	for i in 1:length(vc)
		SimulateMVN!(z, vc[i])
		#add the effects of each variance component
		total_variance += z
	end

	return total_variance
end

#multiple LMM traits

#without computing mean from dataframe and formulas i.e given an evaluated matrix of means
function LMM_trait_simulation(mu, vc::Vector{VarianceComponent})
	n_people = size(mu)[1]
	n_traits = size(mu)[2]

	#preallocate memory for the returned dataframe simulated_trait
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)

	#using the function above to write over the allocated z and simulated_trait
	Aggregate_VarianceComponents!(z, simulated_trait, vc)
	#for each trait add the mean --> MN(mu, Sigma)
	simulated_trait += mu

	out = DataFrame(simulated_trait)

	out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

	return out
end