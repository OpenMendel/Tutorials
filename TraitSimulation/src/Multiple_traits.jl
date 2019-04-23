using DataFrames
using LinearAlgebra
using TraitSimulation
using Random

#in context of linear mixed models and multiple correlated traits, the outcome must be multivariate normal with 
#GRM and environmental variance matrix or constant (iid) B
 
function multiple_trait_simulation(formulas, dataframe, A, B, GRM)
# for multiple traits
#find the number of traits
n_traits = length(formulas)

mean = Vector{Vector{Float64}}(undef, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[i] = mean_formula(formulas[i], dataframe)
end

#concatenate them together
meanvector = vcat(mean...) # take all of the i's and splat them into the meanvector.

#A = cov(hcat(mean...)) # me assuming that we dont know this A and I have to assume it internally
term1 = kron(A, GRM) 
term2 = kron(B, Matrix{Float64}(I, size(GRM)))
Σ = term1 + term2

model = MvNormal(meanvector, Σ)
out1 = rand(model)

out2 = DataFrame(reshape(out1, (size(GRM)[1], n_traits)))

out2 = names!(out2, [Symbol("trait$i") for i in 1:n_traits])

return out2

end 


####
#version 2 

function multiple_trait_simulation2(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholA = cholesky(A)
	cholK = cholesky(GRM)
	cholB = cholesky(B)

	chol_AK = kron(cholA.L, cholK.L)
	chol_BI = kron(cholB.L, Diagonal(ones(n_people)))

	#generate from standard normal
	z_1 = randn(n_people*n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition
	u_1 = chol_AK' * z_1

#second matrix vector mult
	v_1 = chol_AK * u_1

	#generate from standard normal
	z_2 = randn(n_people*n_traits)

#for second variance component
	u_2 = chol_BI' * z_2

	v_2 = chol_BI * u_2


simulated_trait = reshape(v_1 + v_2, (n_people, n_traits))

#now that we have simulated from mvn(0, Sigma)
#we need to add back the mean

mean = Matrix{Float64}(undef, n_people, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[:, i] = mean_formula(formulas[i], dataframe)
end

simulated_trait += mean

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end

##

#version 3

function multiple_trait_simulation3(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholA = cholesky(A)
	cholK = cholesky(GRM)
	cholB = cholesky(B)

	#chol_AK = kron(cholA.L, cholK.L)
	#chol_BI = kron(cholB.L, Diagonal(ones(n_people)))

	#generate from standard normal
	z_1 = randn(n_people, n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition
	#u_1 = chol_AK' * z_1
	new_u1 = cholK.U * z_1 * cholA.L

#second matrix vector mult
	#v_1 = chol_AK * u_1
	new_v1 = cholK.L * new_u1 * cholA.U

	#generate from standard normal
	z_2 = randn(n_people, n_traits)

#for second variance component
	#u_2 = chol_BI' * z_2
	new_u2 = z_2 * cholB.U #identity goes away

	#v_2 = chol_BI * u_2
	new_v2 = new_u2 * cholB.L #identity goes away

#simulated_trait = reshape(new_v1 + new_v2, (n_people, n_traits))
simulated_trait = new_v1 + new_v2

#now that we have simulated from mvn(0, Sigma)
#we need to add back the mean

mean = Matrix{Float64}(undef, n_people, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[:, i] = mean_formula(formulas[i], dataframe)
end

simulated_trait += mean

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end

#version 4 tryiign to make memory allocation better by overwriting 
#variance component matrix 

function multiple_trait_simulation4(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholA = cholesky(A)
	cholK = cholesky(GRM)
	cholB = cholesky(B)

#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)

	#generate from standard normal
	z_1 = randn(n_people, n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition

	mul!(simulated_trait, cholK.U, z_1)
	rmul!(simulated_trait, cholA.L)

#second matrix vector mult

	rmul!(simulated_trait, cholA.U)
	lmul!(cholK.L, simulated_trait) #multiply on left and save to simulated_trait

	#generate from standard normal
	z_2 = randn(n_people, n_traits)

#for second variance component

	mul!(temp, z_2, cholB.U) 
	rmul!(temp, cholB.L)
	simulated_trait += temp

#for each trait
for i in 1:n_traits
	#calculate the mean vector
	simulated_trait[:, i] += mean_formula(formulas[i], dataframe)
end

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end

## scaling to more than 2 variance components 
# make generic kron(A1, B1) + kron(A2, B2) structure for cov matrices
#version 5
function multiple_trait_simulation5(formulas, dataframe, A, B)
	#isposdef(A) cholesky decomp will fail if any A, B not semipd
	#isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(B[1], 1)
	n_traits = size(A[1], 1)

#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)
z = Matrix{Float64}(undef, n_people, n_traits)

for i in 1:length(A)
cholA = cholesky(A[i])
cholB = cholesky(B[i]) #for the ith covariance matrix (VC) in B

	#generate from standard normal
	randn!(z)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition

#need to find which will be CholA, CholB 
	lmul!(cholB.U, z)
	rmul!(z, cholA.L)

#second matrix vector mult
	rmul!(z, cholA.U)
	lmul!(cholB.L, z) #multiply on left and save to simulated_trait

simulated_trait += z
end

#for each trait
for i in 1:n_traits
	#calculate the mean vector
	simulated_trait[:, i] += mean_formula(formulas[i], dataframe)
end

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out
end

# New version using the created VarianceComponent type as an input rather than the manual A, B
#vc = [VarianceComponent(X1[i], Y1[i]) for i in 1:length(X1)]
#vc = [VarianceComponent(X1[1], Y1[1])]

# change this to return a vector of VarianceComponent objects
#this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop

struct VarianceComponent
	A::Matrix{Float64}
	B::Matrix{Float64}
	CholA::Cholesky{Float64,Array{Float64,2}}
	CholB::Cholesky{Float64,Array{Float64,2}}
	function VarianceComponent(A, B) #inner constructor given A, B 
		return(new(A, B, cholesky(A), cholesky(B))) # can construct these
	end
end




function multiple_trait_simulation6(formulas, dataframe, vc::Vector{VarianceComponent})
	#isposdef(A) cholesky decomp will fail if any A, B not semipd
	n_people = size(dataframe, 1)
	n_traits = length(formulas)
#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)
z = Matrix{Float64}(undef, n_people, n_traits)

for i in 1:length(vc)
cholA = vc[i].CholA
cholB = vc[i].CholB #for the ith covariance matrix (VC) in B

	#generate from standard normal
	randn!(z)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition

#need to find which will be CholA, CholB 
	lmul!(cholB.U, z)
	rmul!(z, cholA.L)

#second matrix vector mult
	rmul!(z, cholA.U)
	lmul!(cholB.L, z) #multiply on left and save to simulated_trait

simulated_trait += z
end

#for each trait
for i in 1:n_traits
	#calculate the mean vector
	simulated_trait[:, i] += mean_formula(formulas[i], dataframe)
end

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out
end

#####

#multiple LMM traits

#without computign mean from dataframe and formulas 
function LMM_trait_simulation(mu, vc::Vector{VarianceComponent})
	#isposdef(A) cholesky decomp will fail if any A, B not semipd
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)
z = Matrix{Float64}(undef, n_people, n_traits)

for i in 1:length(vc)
cholA = vc[i].CholA
cholB = vc[i].CholB #for the ith covariance matrix (VC) in B

	#generate from standard normal
	randn!(z)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition

#need to find which will be CholA, CholB 
	lmul!(cholB.U, z)
	rmul!(z, cholA.L)

#second matrix vector mult
	rmul!(z, cholA.U)
	lmul!(cholB.L, z) #multiply on left and save to simulated_trait

simulated_trait += z
end

#for each trait
simulated_trait += mu

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out
end

#single LMM trait  𝐿𝑣+𝜇.
function LMM_trait_simulation(mu, vc::Matrix{T}) where T
	#isposdef(A) cholesky decomp will fail if any A, B not semipd
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)
z = Matrix{Float64}(undef, n_people, n_traits)

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

#####
# AB is an empty vector of variance components
#Take a term A_1 ⊗ B_1 and add its elements to the vector of variance components
#AB = :(VarianceComponent[]) 

# g() = nothing

function append_terms!(AB, summand)
  # elements in args are symbols,
  A_esc = esc(summand.args[2])
  B_esc = esc(summand.args[3])
  push!(AB.args, :(VarianceComponent($A_esc, $B_esc)))
end

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