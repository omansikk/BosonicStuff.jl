# Bibliography:
#  for the constant N indexing:
#    Streltsov et al., 'General mapping for bosonic and fermionic operators in Fock space', doi:10.1103/PhysRevA.81.022124
#  for the constant N basis next! function, although that is no longer in use
#    J. Zhang et al., 'Exact diagonalization: the bose–hubbard model as an example', doi:10.1088/0143-0807/31/3/016

import Base: getindex, length, iterate

# definitions of the basis types

abstract type Basis end

mutable struct Basis_constant_N <: Basis
    L::Integer
    N::Integer
end


mutable struct Basis_global_max_N <: Basis
    L::Integer
    N::Integer
end


mutable struct Basis_local_max_N <: Basis
    L::Integer
    N::Integer
end


mutable struct Basis_composite <: Basis
    bases::Vector{Basis}
    L::Int64
    N::Int64
end


Basis_composite(v::Vector{Basis}) = Basis_composite(v, sum([v[i].L for i in 1:length(v)]), max[v[i].N for i in 1:length(v)])

# definitions the Basis objects as sort of abstract vectors, and allowing iteration

"""
`dimension(basis::Basis)`

Compute the dimension, i.e., the number of basis vectors in `basis`.
"""
dimension(basis::Basis_constant_N) = dimension_constant_N(basis.L, basis.N)
dimension(basis::Basis_global_max_N) = dimension_global_max_N(basis.L, basis.N)
dimension(basis::Basis_local_max_N) = dimension_local_max_N(basis.L, basis.N)


"""
`length(basis::Basis) = dimension(basis::Basis)`

Compute the dimension, i.e., the number of basis vectors in `basis`.
"""
length(basis::Basis) = dimension(basis::Basis)

"""
`find_vector(basis::Basis, index)`

Find the Fock vector corresponding to `index`.
"""
find_vector(basis::Basis_constant_N, index) = find_vector_constant_N(basis.L, basis.N, index)
find_vector(basis::Basis_global_max_N, index) = find_vector_global_max_N(basis.L, basis.N, index)
find_vector(basis::Basis_local_max_N, index) = find_vector_local_max_N(basis.L, basis.N, index)

"""
`getindex(basis::Basis, index) = find_vector(basis::Basis, index)`

Find the Fock vector corresponding to `index`.
"""
getindex(basis::Basis, index) = find_vector(basis, index)


function iterate(basis::Basis, index = 1)
    if index <= length(basis)
        return getindex(basis, index), index + 1
    else
        return nothing
    end
end


#functions for constant N bases

"""
`next!(fock)`

Return the next Fock vector in order in the constant boson number basis. Probably useless now that the bases are iterable.
"""
function next!(fock)
    if length(fock) > 1
        N = sum(fock)
        nk = 1
        for j in reverse(1:length(fock) - 1)
            if fock[j] != 0
                nk = j
                break
            end
        end

        if fock[nk] > 0
            fock[nk] -= 1
            fock[nk + 1] = N - sum(@view fock[1:nk])
            fock[nk + 2:end] .*= 0
        else
            fock[:] .*= 0
            fock[1] = N
        end
    end

    return nothing
end


function dimension_constant_N(L, N)
    d = 1.
    for i in 1:min(N, L - 1)
        d *= (L + N - i) / i
    end

    return round(Int64, d)
end


find_index(basis::Basis_constant_N, fock::Vector{Int64}) = find_index_constant_N(fock)

function find_index_constant_N(fock)
    L = length(fock)
    N = sum(fock)
    index = 1
    for k in 1:L - 1
        index += binomial(N + L - 1 - k - sum(fock[1:k]), L - k)
    end

    return index
end


function find_vector_constant_N(L, N, index)
    fock = zeros(Int64, L)
    i = index; j = 0
    for l in 1:L - 1
        for n in 0:N
            fock[l] = n
            j = binomial(L + N - 1 - l - sum(fock[1:l]), L - l)
            if i > j
                i -= j
                break
            end
        end

        if l == L - 1
            fock[L] = N - sum(fock)
        end

        if sum(fock) == N
            break
        end
    end

    return fock
end


# functions for global max N bases

function dimension_global_max_N(L, N)
    dim = 1
    for n in 1:N
        dim += dimension_constant_N(L, n)
    end

    return dim
end


find_index(basis::Basis_global_max_N, fock) = find_index_global_max_N(fock)

function find_index_global_max_N(fock)
    L = size(fock, 1)
    N = sum(fock)
    if N > 0
        index = 2
        for i in 1:N - 1
            index += dimension_constant_N(L, i)
        end

        for k in 1:L - 1
            index += binomial(N + L - 1 - k - sum(fock[1:k]), L - k)
        end

        return index
    else

        return 1
    end
end


function find_vector_global_max_N(L, N, index)
    i = index
    for n in 0:N
        dim = dimension_constant_N(L, n)
        if i - dim <= 0
            return find_vector_constant_N(L, n, i)
        else
            i -= dim
        end
    end
end

# fuctions for local max N bases

function dimension_local_max_N(L, N)
    return (N + 1)^L
end


find_index(basis::Basis_local_max_N, fock) = find_index_local_max_N(basis.L, basis.N, fock)

function find_index_local_max_N(L, N, fock)
    index = 1
    for l in 1:L
        index += fock[l] * (N + 1)^(L - l)
    end

    return index
end


function find_vector_local_max_N(L, N, index)
    i = index
    fock = zeros(Int64, L)
    for l in 1:L
        m = 0
        for n in N:-1:0
            m = n * (N + 1)^(L - l)
            if m < i
                fock[l] = n
                break
            end
        end

        i -= m
    end

    return fock
end


#functions for composite bases

function dimension(basis::Basis_composite)
    dim = 1
    for b in basis.bases
        dim *= dimension(b)
    end
    
    return dim
end


function find_index(basis::Basis_composite, fock)
    ls = zeros(Int64, length(basis.bases) + 1)
    ls[1] = 1
    for i in 2:length(ls)
        ls[i] = ls[i - 1] + basis.bases[i - 1].L
    end
    
    dims = dimension.(basis.bases)
    index = 1
    for i in 1:length(basis.bases)
        multiplier = i < length(basis) ? prod(dims[i + 1:end]) : 1
        index += (find_index(basis.bases[i], fock[ls[i]:ls[i + 1] - 1]) - 1) * multiplier
    end

    return index
end
    

function find_vector(basis::Basis_composite, index)
    Ls = [basis.bases[i].L for i in 1:length(basis.bases)]
    focks = zeros.(Int64, Ls)
    dims = dimension.(basis.bases)
    for i in 1:length(basis.bases) - 1
        j = 1
        while index > prod(dims[i + 1:end])
            j += 1; index -= prod(dims[i + 1:end])
        end
        
        focks[i] .= find_vector(basis.bases[i], j)
    end

    focks[end] .= find_vector(basis.bases[end], index)
    return reduce(vcat, focks)
end


# functions for initialising product states

"""
`product_state(basis::Basis, fock::Vector{Int64})`

`product_state(basis::Basis, fock::Tuple{Int64, Int64})`

`product_state(basis::Basis, fock::Vector{Tuple{Int64, Int64}})`

Return the product state `fock` represented in the basis `basis`.

E.g. the calls `product_state(basis, [0, 3, 0, 0])` and `product_state(basis, (3, 2))` return the same vector.

Similarly, `product_state(basis, [0, 3, 0, 2, 0])` and `product_state(basis, [(3, 2), (2, 4)])` result in the same vector.
"""
function product_state(basis::Basis, fock::Vector{Int64})
    state = zeros(ComplexF64, length(basis))
    state[find_index(basis, fock)] = 1.

    return state
end

function product_state(basis::Basis, fock::Tuple{Int64, Int64})
    state = zeros(ComplexF64, length(basis))
    vector = zeros(Int64, basis.L)
    vector[fock[2]] = fock[1]
    state[find_index(basis, vector)] = 1.

    return state
end

function product_state(basis::Basis, fock::Vector{Tuple{Int64, Int64}})
    state = zeros(ComplexF64, length(basis))
    vector = zeros(Int64, basis.L)
    for i in 1:length(fock)
        vector[fock[i][2]] = fock[i][1]
    end

    state[find_index(basis, vector)] = 1.

    return state
end

"""
print_State(basis::Basis, state; cutoff = 0.99)

Print `state` prettily in descending order of the absolute value of the components.
"""
function print_state(basis::Basis, state; cutoff = 0.99)
    sorting_thing = []
    print_basis = []
    for i in 1:length(basis)
        push!(sorting_thing, abs(state[i]))
        fock = "|" * prod(string.(basis[i])) * "⟩"
        push!(print_basis, fock)
    end

    indices = sortperm(sorting_thing, rev = true)
    total = 0.
    for i in 1:length(basis)
        total += abs2(state[indices[i]])
        println(round(state[indices[i]], digits = 4), " ", print_basis[indices[i]], "         ", round(total, digits = 4))
        if total > cutoff
            break
        end
    end

    println()

    return nothing
end



