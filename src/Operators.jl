using SparseArrays, LinearAlgebra

"""
`operator(basis::Basis, action::Function; element_type = Float64, add_adjoint = false)`

Construct a sparse matrix representation (`SparseMatrixCSC{element_type, Int64}`) of an operator whose action on the basis vectors `basis[i]` is defined by the function `action(fock)`. The function `action` should take as its argument a basis vector and return one or more basis vectors and corresponding matrix elements:

`fock, element = action(basis[i])`,

where `typeof(fock)` is `Vector{Int64}` or `Vector{Vector{Int64}}` and `typeof(element)` is `element_type` or `Vector{element_type}`.

Note that the function assumes that the output of `action` actually belongs to `basis`. If it does not, the function either breaks or works incorrectly.
"""
function operator(basis::Basis, action::Function; element_type = Float64, add_adjoint = false)
    dim = length(basis)
    op = spzeros(element_type, dim, dim)
    for i in 1:dim
        fock, element = action(basis[i])
        if length(fock) != 0
            if isa(fock[1], Vector)
                for j in 1:length(fock)
                    if element[j] != 0
                        op[find_index(basis, fock[j]), i] += element[j]
                    end
                end
            elseif element != 0
                op[find_index(basis, fock), i] += element
            end
        end
    end

    if add_adjoint == true
        op .+= op'
    end

    return op
end


"""
`operator(basis::Basis, action::Function; element_type = Float64, add_adjoint = false)`

Construct a vector representation of a diagonal operator whose action on the basis vectors `basis[i]` is defined by the function `action(fock)`. The function `action` should take as its argument a basis vector and return the corresponding matrix element:

element = action(basis[i])`,

where `typeof(element)` is `element_type`.
"""
function diagonal_operator(basis::Basis, action::Function; element_type = Float64)
    dim = length(basis)
    op = zeros(element_type, dim)
    for i in 1:dim
        op[i] += action(basis[i])
    end

    return op
end


"""
`number(basis::Basis, site::Int64)`

Construct the number operator ``\\hat{n}_\\ell`` on the site `site` in the basis `basis`.
"""
function number(basis::Basis, site::Int64)
    return operator(basis, fock -> (fock, fock[site]))
end


"""
numbers(basis::Basis)

Create a vector of all the local number operators in the basis `basis`. See `number(basis, site)`
"""
function numbers(basis::Basis)
    return [number(basis, l) for l in 1:basis.L]
end


"""
`annihilation(basis::Basis, site::Int64)`

Construct the annihilation operator ``\\hat{a}_\\ell`` on the site `site` in the basis `basis`.
"""
function annihilation(basis::Basis, site::Int64)
    return operator(basis, fock ->

    begin
        fock_out = copy(fock)
        if fock[site] > 0
            fock_out[site] -= 1

            return fock_out, sqrt(fock[site])
        else
            return fock, 0
        end
    end
    )
end


"""
`annihilations(basis::Basis, site::Int64)`

Create a vector of all the local annihilation operators in the basis `basis`. See `annihilation(basis, site)`
"""
function annihilations(basis::Basis)
    return [annihilation(basis, l) for l in 1:basis.L]
end


"""
`hops(basis::Basis, fock; periodic = false)`

Return the basis vectors in `basis` coupled to `fock` and the corresponding matrix elements. That, is the action of the hopping part
``
\\hat{H}_J = \\sum_\\ell (\\hat{a}^\\dagger_\\ell \\hat{a}_{\\ell + 1} + \\text{H.c.})
``
of the ABH Hamiltonian; see `ABH_Hamiltonian`.
"""
function hops(basis::Basis, fock; periodic = false)
    focks_out::Vector{Vector{Int64}} = []
    elements::Vector{Float64} = []
    copy_fock = copy(fock)
    for l in 1:basis.L - 1
        if fock[l] > 0
            copy_fock .= copy(fock)
            copy_fock[l] -= 1
            copy_fock[l + 1] += 1
            if find_index(basis, fock) <= length(basis)
                push!(focks_out, copy(copy_fock))
                push!(elements, sqrt(fock[l] * copy_fock[l + 1]))
            end
        end

    end

    if periodic == true && fock[basis.L] > 0
        copy_fock .= copy(fock)
        copy_fock[L] -= 1
        copy_fock[1] += 1
        if find_index(basis, fock) <= length(basis)
            push!(focks_out, copy(copy_fock))
            push!(elements, sqrt(fock[l] * copy_fock[l + 1]))
        end
    end

    return focks_out, elements
end


# should be easy to cut the below to about a third of the length

"""
`hops(basis::Basis, fock; periodic = 0)`

Return the basis vectors in `basis` coupled to `fock` and the corresponding matrix elements. That, is the action of the hopping part
``
\\hat{H}_J = \\sum_\\ell (\\hat{a}^\\dagger_\\ell \\hat{a}_{\\ell + 1} + \\text{H.c.})
``

of the 2D rectangulat ABH Hamiltonian; see `ABH_Hamiltonian_2D`.

The variable `periodic` determines how many of the edges of the array are periodic.
"""
function hops_2D(basis::Basis, fock, L1, L2; periodic = 0)
    focks_out::Vector{Vector{Int64}} = []
    elements::Vector{Float64} = []
    if periodic == 0
        for k in 1:L2
            for j in 1:L1
                site = j + (k - 1) * L1
                if fock[site] > 0
                    if j < L1
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + 1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    end

                    if k < L2
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + L1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    end
                end
            end
        end
    elseif periodic == 1
        for k in 1:L2
            for j in 1:L1
                site = j + (k - 1) * L1
                if fock[site] > 0
                    if j < L1
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + 1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    else
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + 1 - L1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    end

                    if k < L2
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + L1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    end
                end
            end
        end
    elseif periodic == 2
        for k in 1:L2
            for j in 1:L1
                site = j + (k - 1) * L1
                if fock[site] > 0
                    if j < L1
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + 1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    else
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + 1 - L1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    end

                    if k < L2
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[site + L1] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    else
                        copy_fock .= copy(fock)
                        copy_fock[site] -= 1
                        copy_fock[j] += 1
                        if find_index(basis, fock) <= length(basis)
                            push!(focks_out, copy_fock)
                            push!(elements, sqrt(fock[site] * focks_out[end][site + 1]))
                        end
                    end
                end
            end
        end
    end

    return focks_out, elements
end


u(fock) = -0.5 * sum(fock.^2 .- fock)

"""
`ABH_Hamiltonian(basis::Basis; periodic = false, split = true, J = 1, U = 1, dims = (0, 0))`

Construct a sparse matrix representation of the attractive Bose-Hubbard Hamiltonian
``
\\hat{H} = -\\frac{U}{2}\\sum_\\ell \\hat{n} (\\hat{n}_\\ell - 1) + J \\sum_\\ell (\\hat{a}^\\dagger_\\ell \\hat{a}_{\\ell + 1} + \\text{H.c.}).
``

The interaction and hopping terms are returned separately by default (`split = true`).

For a rectangular 2D array, set `dims = (L1, L2)`, where `L1` and `L2` are the lengths of the edges. Set `periodic` to 1 for a periodic array in one direction, and 2 for both directions.
"""
function ABH_Hamiltonian(basis::Basis; periodic = false, split = true, J = 1, U = 1, dims::Tuple{Int64, Int64} = (0, 0))
    HU = operator(basis, fock -> (fock, u(fock))) .* U
    if dims == (0, 0)
        HJ = operator(basis, fock -> hops(basis, fock, periodic = periodic), add_adjoint = true) .* J
    else
        if periodic == false
            HJ = operator(basis, fock -> hops_2D(basis, fock, dims[1], dims[2], periodic = 0), add_adjoint = true) .* J
        else
            HJ = operator(basis, fock -> hops_2D(basis, fock, dims[1], dims[2], periodic = periodic), add_adjoint = true) .* J
        end
    end

    if split == true
        return HU, HJ
    else
        return HU .+ HJ
    end
end


"""
`number_disorder(basis::Basis; pattern = 2. .* rand(basis.L) .- 1.)`

Construct a sparse matrix representation of on-site disorder
``
\\hat{H}_{\\omega} = \\sum_\\ell \\omega_\\ell \\hat{n}_\\ell.
``
"""
function number_disorder(basis::Basis; pattern = 2. .* rand(basis.L) .- 1.)
    return operator(basis, fock -> (fock, sum(pattern .* fock)))
end


"""
anharmonicity_disorder(basis::Basis; pattern = 2. .* rand(basis.L) .- 1.)`

Construct a sparse matrix representation of on-site anharmonicity disorder
``
\\hat{H}_{\\delta U} = -\\frac{1}{2}\\sum_\\ell \\delta U_\\ell \\hat{n}_\\ell (\\hat{n}_\\ell - 1).
``
"""
function anharmonicity_disorder(basis::Basis; pattern = 2. .* rand(basis.L) .- 1)
    return operator(basis, fock -> (fock, -0.5 * sum(pattern .* (fock.^2 .- fock))))
end
