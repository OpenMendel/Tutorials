"""
This module implements computation of kinship and other identity coefficients.
The kinship matrix is computed recursively by the standard formula.
The delta7 matrix is derived from the kinship matrix for non-inbred pedigrees.
The remaining identity coefficients of Jacquard are computed by simulation.
"""
module MendelKinship
#
# Required OpenMendel packages and modules.
#
using MendelBase
using SnpArrays
#
# Required external modules.
#
using PlotlyJS
using StatsBase
using DataFrames                        # From package DataFrames.
##
using CSV
##

export Kinship

"""
This is the wrapper function for the Kinship analysis option.
"""
function Kinship(control_file = ""; args...)
#
  const KINSHIP_VERSION :: VersionNumber = v"0.2.0"
#
# Print the logo. Store the initial directory.
#
  print(" \n \n")
  println("     Welcome to OpenMendel's")
  println("     Kinship analysis option")
  println("        version ", KINSHIP_VERSION)
  print(" \n \n")
  println("Reading the data.\n")
  initial_directory = pwd()
#
# The user specifies the analysis option via a set of keywords.
# Start the keywords at their default values.
#
  keyword = set_keyword_defaults!(Dict{AbstractString, Any}())
#
# Keywords unique to this analysis should be first defined here.
# Set their default values using the format:
# keyword["some_keyword_name"] = default_value
#
  keyword["repetitions"] = 1
  keyword["xlinked_analysis"] = false
  keyword["kinship_output_file"] = "kinship_file_output.txt"
  keyword["compare_kinships"] = false
  keyword["maf_threshold"] = 0.01
  keyword["grm_method"] = "MoM" # MoM is less sensitive to rare snps
  keyword["deviant_pairs"] = 0
  keyword["kinship_plot"] = ""
  keyword["z_score_plot"] = ""
#
# Process the run-time user-specified keywords that will control the 
# analysis. This will also initialize the random number generator.
#
  process_keywords!(keyword, control_file, args)
#
# Check that the correct analysis option was specified.
#
  lc_analysis_option = lowercase(keyword["analysis_option"])
  if (lc_analysis_option != "" &&
      lc_analysis_option != "kinship")
     throw(ArgumentError(
       "An incorrect analysis option was specified.\n \n"))
  end
  keyword["analysis_option"] = "Kinship"
#
# Read the genetic data from the external files named in the keywords.
#
  (pedigree, person, nuclear_family, locus, snpdata, locus_frame, 
    phenotype_frame, pedigree_frame, snp_definition_frame) =
    read_external_data_files(keyword)
#
# Execute the specified analysis.
#
  println(" \nAnalyzing the data.\n")
  if keyword["compare_kinships"]
    @assert snpdata.snps != 0 "Need Snp data to compare kinships"
    @assert snpdata.snps == length(unique(snpdata.snpid)) "non-unique snp ids"
    @assert person.people == length(unique(person.name)) "non-unique person names"
#
# Find each person's order on data entry before MendelBase permutes them.
# 
    entryorder = convert(Vector{Int64}, pedigree_frame[:EntryOrder])
#
# Calculate empiric and theoretical kinships and append fisher's z-scores 
# of these to the kinship frame.
#
    kinship_frame = compare_kinships(pedigree, person, snpdata, keyword, entryorder)
    kinship_frame = compute_fishers_z(kinship_frame)
#
# Create and save plots based on user's requests.
#
    name = make_plot_name(kinship_frame) 
    if keyword["kinship_plot"] != "" 
      my_compare_plot = make_compare_plot(kinship_frame, name)
      PlotlyJS.savefig(my_compare_plot, keyword["kinship_plot"] * ".html")
      println("Kinship plot saved.")
    end
    if keyword["z_score_plot"] != ""
      my_fisher_plot = plot_fisher_z(kinship_frame, name)
      PlotlyJS.savefig(my_fisher_plot, keyword["z_score_plot"] * ".html")
      println("Fisher's plot saved.")
    end
  else
#
# Otherwise, simply compute theoretical coefficients.
#
    kinship_frame = kinship_option(pedigree, person, keyword)
  end
#
# Save the kinship frame and write it to the designated file and REPL.
#
  name = convert(String, keyword["kinship_output_file"])
  CSV.write(name, kinship_frame)
  display(kinship_frame)
#
# Finish up by closing, and thus flushing, the output file.
# Return to the initial directory.
#
  println(" \n \nMendel's analysis is finished.\n")
  close(keyword["output_unit"])
  cd(initial_directory)
  return nothing
end # function Kinship

"""Compares theoretical and empirical kinship coefficients."""

function compare_kinships(pedigree::Pedigree, person::Person,
  snpdata::SnpData, keyword::Dict{AbstractString, Any}, entryorder::Vector{Int})
#
  people = person.people
  pedigrees = pedigree.pedigrees
  xlinked = keyword["xlinked_analysis"]
  maf_threshold = keyword["maf_threshold"]
  deviant_pairs = keyword["deviant_pairs"]
  if deviant_pairs == 0
    deviant_pairs = people^2
  end
#
# Create a vector of kinship matrices, one for each pedigree.
#
  kinship = Vector{Matrix{Float64}}(pedigrees)
##
## Read the full pedigree, including people with no snp information 
##
  # ped_full = CSV.read("Ped10cextra.csv")
#
# Create a copy of snpdata.personid that orders people similarly to person.name.
# People are permuted so the parents come before children. This facilitates
# computation of theoretical kinships. In SnpArrays people are not permuted.
#
  personid_in_snpdata = copy(snpdata.personid)
  ipermute!(personid_in_snpdata, entryorder)
#
# Since people are reordered, compute identification maps to match
# positions of ids in two vectors of ids.
#     
  name_to_id = indexin(person.name, personid_in_snpdata)
  id_to_name = indexin(personid_in_snpdata, person.name)
#
# Compute and transform the genetic relationship matrix.
#
  if keyword["grm_method"] == "GRM"
    GRM = grm(snpdata.snpmatrix, method=:GRM, maf_threshold=maf_threshold)
  else
    GRM = grm(snpdata.snpmatrix, method=:MoM)
  end
  clamp!(GRM, -0.99999, 0.99999) 
  map!(x -> atanh(x), GRM, GRM) #fisher's transformation
#
# Loop over all pedigrees.
#
  for ped = 1:pedigrees
#
# Compute theoretical coefficients and adjust the GRM matrix.
#
    kinship[ped] = kinship_matrix(pedigree, person, ped, xlinked)
    q = pedigree.start[ped] - 1
    for j = 1:pedigree.individuals[ped]
      jj = name_to_id[j + q]
      if jj == 0; continue; end
      for i = j:pedigree.individuals[ped]
        ii = name_to_id[i + q]
        if ii == 0; continue; end
        GRM[ii, jj] = GRM[ii, jj] - atanh(kinship[ped][i, j])
        GRM[jj, ii] = GRM[ii, jj]
      end
    end
  end
#
# Find the indices of the most deviant pairs.
#
  p = selectperm(vec(GRM), 1:deviant_pairs, by = abs, rev = true)
  (rowindex, columnindex) = ind2sub((people, people), p)
#
# Enter the most deviant pairs in a data frame.
#
  kinship_frame = DataFrame(Pedigree1 = String[], Pedigree2 = String[], 
    Person1 = String[], Person2 = String[], theoretical_kinship= Float64[], 
    empiric_kinship=Float64[])
  r = 0.0
  for k = 1:deviant_pairs
    (ii, jj) = (rowindex[k], columnindex[k]) # index of kth largest deviation
    (i, j) = (id_to_name[ii], id_to_name[jj]) # maps snpid back to person name
    if j > i; continue; end
    (pedi, pedj) = (person.pedigree[i], person.pedigree[j]) # pedigrees of i and j    
    if pedi == pedj
      q = pedigree.start[pedi] - 1
      r = GRM[ii, jj] + atanh(kinship[pedi][i - q, j - q]) 
      push!(kinship_frame, [pedigree.name[pedi], pedigree.name[pedj], person.name[i], 
        person.name[j], kinship[pedi][i - q, j - q], tanh(r)])     
    else
      push!(kinship_frame, [pedigree.name[pedi], pedigree.name[pedj], person.name[i], 
        person.name[j], 0.0, tanh(GRM[ii, jj])])
    end
  end
  return kinship_frame
end

"""
This function orchestrates the computation via gene dropping
of the Kinship and Delta7 coefficients deterministically
and Jacquard's 9 identity coefficients stochastically.
The results are placed in a dataframe.
"""
function kinship_option(pedigree::Pedigree, person::Person,
  keyword::Dict{AbstractString, Any})

  pedigrees = pedigree.pedigrees
  repetitions = keyword["repetitions"]
  xlinked = keyword["xlinked_analysis"]
  #
  # Define dataframes for theoretical Kinship, Delta7, and Jacquard 
  # coefficients.
  #
    # kinship_frame = DataFrame(Pedigree1 = String[], Pedigree2 = String[], 
    # Person1 = String[], Person2 = String[], theoretical_kinship= Float64[], 
    # empiric_kinship=Float64[])
  kinship_dataframe = DataFrame(
    ped1 = Int[], Pedigree = String[], Person1 = String[], Person2 = String[], Kinship = Float64[])
  if !xlinked
    delta_dataframe = DataFrame(ped2 = Int[], Pedigree = String[], Person1 = String[], 
      Person2 = String[], Delta7 = Float64[])
  end
  coefficient_dataframe = DataFrame(
    ped3 = Int[], Pedigree = String[], Person1 = String[], Person2 = String[],
    delta1 = Float64[], delta2 = Float64[], delta3 = Float64[],
    delta4 = Float64[], delta5 = Float64[], delta6 = Float64[],
    delta7 = Float64[], delta8 = Float64[], delta9 = Float64[])
  #
  # Loop over all pedigrees.
  #
  for ped = 1:pedigrees
    #
    # Retrieve the various coefficients.
    #
    kinship = kinship_matrix(pedigree, person, ped, xlinked)
    if !xlinked
      delta7 = delta7_matrix(pedigree, person, kinship, ped)
    end
    coefficient = jacquard_coefficients(pedigree, person, ped,
      repetitions, xlinked)
    #
    # Insert the computed coefficients into the appropriate dataframes.
    #
    q = pedigree.start[ped] - 1
    for i = 1:pedigree.individuals[ped]
      for j = i:pedigree.individuals[ped]
        push!(kinship_dataframe, [ped, pedigree.name[ped],
          person.name[i + q], person.name[j + q], kinship[i, j]])
        if !xlinked
          push!(delta_dataframe, [ped, pedigree.name[ped],
            person.name[i + q], person.name[j + q], delta7[i, j]])
        end
        push!(coefficient_dataframe, [ped, pedigree.name[ped],
          person.name[i + q], person.name[j + q],
          coefficient[1, i, j], coefficient[2, i, j], coefficient[3, i, j],
          coefficient[4, i, j], coefficient[5, i, j], coefficient[6, i, j],
          coefficient[7, i, j], coefficient[8, i, j], coefficient[9, i, j]])
      end
    end
  end
  #
  # Join the separate dataframes, and sort the combined dataframe so
  # that relative pairs are lumped by pedigree.
  #
  if xlinked
    combined_dataframe = join(kinship_dataframe, coefficient_dataframe,
      on = [:Pedigree, :Person1, :Person2])
    sort!(combined_dataframe, cols = [:ped1, :Person1, :Person2])
    delete!(combined_dataframe, [:ped1, :ped3])
  else
    temp_dataframe = join(kinship_dataframe, delta_dataframe,
      on = [:Pedigree, :Person1, :Person2])
    combined_dataframe = join(temp_dataframe, coefficient_dataframe,
      on = [:Pedigree, :Person1, :Person2])
    sort!(combined_dataframe, cols = [:ped1, :Person1, :Person2])
    delete!(combined_dataframe, [:ped1, :ped2, :ped3])
  end
  #
  # Combined coefficient frame to a file and return.
  #
  return combined_dataframe
end # function kinship_option

"""
This function recursively computes a kinship matrix for each pedigree.
Children must appear after parents. Each pedigree occurs in a block
demarked by its start and finish and is defined implicitly by
the provided fathers and mothers. Person i is a pedigree founder
if and only if mother[i] = 0. X-linked inheritance is allowed.
"""
function kinship_matrix(pedigree::Pedigree, person::Person,
  ped::Int, xlinked::Bool)
  #
  # Allocate a kinship matrix and an offset.
  #
  start = pedigree.start[ped]
  finish = pedigree.finish[ped]
  kinship = zeros(pedigree.individuals[ped], pedigree.individuals[ped])
  q = start - 1
  for i = start:finish
    #
    # Initialize the kinship coefficients for founders.
    #
    j = person.mother[i]
    if j == 0
      if xlinked && person.male[i]
        kinship[i - q, i - q] = 1.0
      else
        kinship[i - q, i - q] = 0.5
      end
    else
      #
      # Compute the kinship coefficients of descendants by averaging.
      #
      if xlinked && person.male[i]
        kinship[i - q, i - q] = 1.0
        for m = start:i - 1
          kinship[i - q, m - q] = kinship[j - q, m - q]
          kinship[m - q, i - q] = kinship[i - q, m - q]
        end
      else
        k = person.father[i]
        kinship[i - q, i - q] = 0.5 + 0.5 * kinship[j - q, k - q]
        for m = start:i - 1
          kinship[i - q, m - q] = 0.5 * (kinship[j - q, m - q]
             + kinship[k - q, m - q])
          kinship[m - q, i - q] = kinship[i - q, m - q]
        end
      end
    end
  end
  #
  # Extend the kinship matrix to co-twins.
  #
  cotwin_extension!(kinship, pedigree, person, ped)
  return kinship
end # function kinship_matrix

"""
This function computes the Delta7 matrix for non-inbred pedigrees.
Children must appear after parents. Each pedigree occurs in a block
and is defined implicitly by the provided fathers and mothers.
Person i is a pedigree founder if and only if mother[i] = 0.
"""
function delta7_matrix(pedigree::Pedigree, person::Person,
  kinship::Matrix{Float64}, ped::Int)

  start = pedigree.start[ped]
  finish = pedigree.finish[ped]
  delta7 = zeros(pedigree.individuals[ped], pedigree.individuals[ped])
  q = start - 1
  for i = start:finish
    for j = start:i
      if j == i
        delta7[i - q, i - q] = 1.0
      elseif person.mother[i] > 0 && person.mother[j] > 0
        k = person.father[i]
        l = person.mother[i]
        m = person.father[j]
        n = person.mother[j]
        delta7[i - q, j - q] = kinship[k - q, m - q] * kinship[l - q, n - q] +
          kinship[k - q, n - q] * kinship[l - q, m - q]
        delta7[j - q, i - q] = delta7[i - q, j - q]
      end
    end
  end
  #
  # Extend the delta7 matrix to co-twins.
  #
  cotwin_extension!(delta7, pedigree, person, ped)
  return delta7
end # function delta7_matrix

"""
This function computes Jacquard's nine condensed identity
coefficients for each pedigree by simulation. Children
must appear after parents. A pedigree is defined implicitly
by giving fathers and mothers. Person i is a pedigree founder
if and only if mother[i] = 0. The more replicates requested,
the more precision attained. X-linked inheritance is possible.
"""
function jacquard_coefficients(pedigree::Pedigree, person::Person,
  ped::Int, replicates::Int, xlinked::Bool)

  start = pedigree.start[ped]
  finish = pedigree.finish[ped]
  ped_size = pedigree.individuals[ped]
  coefficient = zeros(9, ped_size, ped_size)
  x = zeros(ped_size, ped_size)
  source = zeros(Int, 2, ped_size)
  #
  # Loop over all replicates.
  #
  q = start - 1
  for rep = 1:replicates
    f = 0
    for j = start:finish
      if person.mother[j] == 0
        f = f + 1
        source[1, j - q] = f
        if xlinked && person.male[j]
        else
          f = f + 1
        end
        source[2, j - q] = f
      else
        if xlinked && person.male[j]
          u = rand(1:2, 1)
          source[1, j - q] = source[u[1], person.mother[j] - q]
          source[2, j - q] = source[1, j - q]
        else
          u = rand(1:2, 2)
          source[1, j - q] = source[u[1], person.father[j] - q]
          source[2, j - q] = source[u[2], person.mother[j] - q]
        end
      end
    end
    #
    # Loop over all pairs.
    #
    for j = start:finish
      for k = start:finish
        l = identity_state(source[:, j - q], source[:, k - q])
        coefficient[l, j - q, k - q] = coefficient[l, j - q, k - q] + 1.0
      end
    end
  end
  #
  # Extend the coefficients to co-twins.
  #
  for i = 1:9
    x = reshape(coefficient[i, :, :], ped_size, ped_size)
    cotwin_extension!(x, pedigree, person, ped)
    coefficient[i, :, :] = reshape(x, 1, ped_size, ped_size)
  end
  return coefficient / replicates
end # function jacquard_coefficients

"""
This function extends identity coefficients to co-twins.
"""
function cotwin_extension!(x::Matrix{Float64}, pedigree::Pedigree,
  person::Person, ped::Int)

  q = pedigree.start[ped] - 1
  for i = pedigree.finish[ped] + 1:pedigree.twin_finish[ped]
    m = person.primary_twin[i]
    for j = pedigree.start[ped]:pedigree.finish[ped]
      x[i - q, j - q] = x[m - q, j - q]
      x[j - q, i - q] = x[i - q, j - q]
    end
    for j = pedigree.finish[ped] + 1:pedigree.twin_finish[ped]
      n = person.primary_twin[j]
      x[i - q, j - q] = x[m - q, n - q]
    end
  end
end # function cotwin_extension!

"""
This function assigns one of the 9 condensed identity states
to two sourced genotypes. The integer n is the number of
unique genes among the 4 sampled genes.
"""
function identity_state(source1::Vector{Int}, source2::Vector{Int})

  n = length(union(IntSet(source1), IntSet(source2)))
  if n == 4
    return 9
  elseif n == 3
    if source1[1] == source1[2]
      return 4
    elseif source2[1] == source2[2]
      return 6
    else
      return 8
    end
  elseif n == 2
    if source1[1] == source1[2]
      if source2[1] == source2[2]
        return 2
      else
        return 3
      end
    else
      if source2[1] == source2[2]
        return 5
      else
        return 7
      end
    end
  else
    return 1
  end
end # function identity_state

"""Assembles names for printing."""

function make_plot_name(x::DataFrame)
  name = Array{String}(size(x, 1))
  for i in 1:length(name)
    name[i] = "Person1=" * x[i, 3] * ", " * "Person2=" * x[i, 4]
  end
  return name
end

"""Composes the comparison plot using PlotlyJS."""

function make_compare_plot(x::DataFrame, name::Vector{String})
  trace1 = scatter(;x=x[:theoretical_kinship], 
    y=x[:empiric_kinship], mode="markers", 
    name="empiric kinship", text=name)
#  
  trace2 = scatter(;x=[1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 0.0],
    y=[1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 0.0], 
    mode="markers", name="marker for midpoint")
#      
  layout = Layout(;title="Compare empiric vs theoretical kinship",
    hovermode="closest", 
    xaxis=attr(title="Theoretical kinship", 
      showgrid=false, zeroline=false),
    yaxis=attr(title="Empiric Kinship", zeroline=false))
#  
  data = [trace1, trace2]
  plot(data, layout)
end

"""Computes Fisher's atanh transform and standardizes it."""

function compute_fishers_z(x::DataFrame)
  theoretical_transformed = map(atanh, x[:theoretical_kinship])
  empiric_transformed = map(atanh, x[:empiric_kinship])
  difference = empiric_transformed - theoretical_transformed
  new_zscore = zscore(difference) #in StatsBase package
  return [x DataFrame(fishers_zscore = new_zscore)]
end

"""Plot the transformed data."""

function plot_fisher_z(x::DataFrame, name::Vector{String})
    trace1 = histogram(x=x[:fishers_zscore], text=name)
    data = [trace1]
#    
    layout = Layout(barmode="overlay", 
        title="Z-score plot for Fisher's statistic",
        xaxis=attr(title="Standard deviations"),
        yaxis=attr(title="count"))
#    
    plot(data, layout)
end
##

end # module MendelKinship

