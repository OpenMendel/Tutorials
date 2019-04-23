using Documenter, TraitSimulation

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs(
    format = Documenter.HTML(),
    sitename = "TraitSimulation",
    modules = [TraitSimulation]
)

deploydocs(
    repo   = "github.com/sarah-ji/TraitSimulation.jl.git", 
    target = "build"
)