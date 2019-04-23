
# The first task is to find the variables to make the mean vector:
find_variables(x) = find_variables!(Symbol[], x) #this is so that we can call the function without the exclamation
function find_variables!(var_names, x::Number) #if the variable name is a number then just return it without doing anything
    return var_names
end
function find_variables!(var_names, x::Symbol) # if the variable name is a symbol then push it to the list of var_names because its a name of a column
    push!(var_names, x)
end

function find_variables!(var_names, x::Expr) # if the variable is a expression object then we have to crawl through each argument of the expression
    # safety checking
    if x.head == :call  # check for + symbol bc we are summing linear combinations within each expression argument
      # pass the remaining expression
      for argument in x.args[2:end] # since the first argument is the :+ call 
        find_variables!(var_names, argument) #recursively find the names given in each argument so check if number, if symbol, if expression etc.again agian
    end
end
return var_names
end

function search_variables!(x::Expr, var::Symbol)
    for i in eachindex(x.args)
        if x.args[i] == var # if the argument is one of the variables given then just put it in the right format df[:x1] 
            x.args[i] = Meta.parse(string(:input_data_from_user,"[", ":", var, "]"))
        elseif x.args[i] isa Expr # else if the argument is an expression (i.e not a varaible (symbol) or a number) then 
            search_variables!(x.args[i], var) #go through this function recursively on each of the arguments of the expression object
        end
    end
    return x
end

function search_variables!(x::Expr, vars...) # this is for when you have more than one variable name found in the string
    for var in vars #goes through each of the variables in the vector vars
        x = search_variables!(x, var) #runs the recursion on each variable in vars
    end
    return x 
end

function mean_formula(user_formula_string::String, df::DataFrame)
    global input_data_from_user = df
    
    users_formula_expression = Meta.parse(user_formula_string)
    if(users_formula_expression isa Expr)
        found_markers = find_variables(users_formula_expression) #store the vector of symbols of the found variables 

        dotted_args = map(Base.Broadcast.__dot__, users_formula_expression.args) # adds dots to the arguments in the expression 
        dotted_expression = Expr(:., dotted_args[1], Expr(:tuple, dotted_args[2:end]...)) #reformats the exprssion arguments by changing the variable names to tuples of the variable names to keep the dot structure of julia

        julia_interpretable_expression = search_variables!(dotted_expression, found_markers...) #gives me the julia interpretable exprsesion with the dataframe provided

        mean_vector = eval(Meta.parse(string(julia_interpretable_expression))) #evaluates the julia interpretable expression on the dataframe provided
    else
        mean_vector = [users_formula_expression for i in 1:size(df, 1)]
    end
    return mean_vector
end

