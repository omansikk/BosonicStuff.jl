# Bibliography:
# Peotta et al., 'Determination of dynamical quantum phase transitions in strongly correlated many-body systems using Loschmidt cumulants', doi:10.1103/PhysRevX.11.041018

using LinearAlgebra, Polynomials

"""
`inverse_participation_ratio(state, numbers)`

Compute the scaled inverse participation ratio.
"""
function inverse_participation_ratio(state, ns)
    out = 0.
    L = size(ns, 1)
    for l in 1:L
        out += abs2(state' * ns[l] * state)

    end

    return (ns[1][1, 1]^2 / out - 1.) / (L - 1)
end


function compute_entropy(schmidt)
    S = svd(schmidt).S
    entropy = 0.0
    for i in 1:sub_dim
        if S[i] > 1e-16
            alpha2 = abs2(S[i])
            entropy += -alpha2 * log(alpha2)
        else
            break
        end
    end

    return entropy
end

"""
`entropy(basis::Basis, state; sites = Integer(ceil(basis.L / 2)))`

Compute the bipartite von Neumann entanglement entropy. Defaults to half a chain (`sites = Integer(ceil(basis.L / 2)))`).
"""
function entropy(basis::Union{Basis_constant_N, Basis_global_max_N}, state; sites = Integer(ceil(basis.L / 2)))
    sub_dim = dimension_global_max_N(sites, N)
    schmidt = zeros(ComplexF64, sub_dim, dimension_global_max_N(L - sites, N))
    for i in 1:length(basis)
        schmidt[find_index_global_max_N(basis[i][1:sites]), find_index_global_max_N(basis_vector[sites + 1: end])] = state[i]
    end

    return compute_entropy(schmidt)
end


function entropy(basis::Basis_local_max_N, state; sites = Integer(ceil(basis.L / 2)))
    sub_dim = dimension_local_max_N(sites, N)
    schmidt = zeros(ComplexF64, sub_dim, dimension_local_max_N(L - sites, N))
    for i in 1:length(basis)
        schmidt[find_index_local_max_N(basis[i][1:sites]), find_index_local_max_N(basis_vector[sites + 1: end])] = state[i]
    end

    return compute_entropy(schmidt)
end


function toeplitz(vals)
    dim = Integer(floor((length(vals) + 1) / 2))
    T = zeros(ComplexF64, dim, dim)
    vals = reverse(vals)
    for i in 1:dim
        T[i, :] = vals[end - i - dim + 1:end - i]
    end

    return T
end


function moments(H, state, n)
    dim = length(state)
    V = zeros(n, dim)
    V[1, :] = H * state
    for i in 2:n
        V[i, :] .= H * V[i - 1, :]
    end

    return V
end

"""
loschmidt_zeros(moments, nlz, time, eps = 1e-2)

Compute the `nlz` Loschmidt zeros closest to `time`.
"""
function loschmidt_zeros(moments, nlz, time, eps = 1e-2)
    cs = copy(moments)
    nc = length(moments)
    for n in 2:nc
        for m in 1:n - 1
            cs[n] -= binomial(n - 1, m - 1) * cs[m] * moments[n - m]
        end
    end

    for n in 1:nc
        cs[n] *= (-1)^(n - 1) / factorial(big(n - 1))
    end

    ks = cs[end - nlz + 1:end]
    T = toeplitz(cs[collect(nc - 2 * nlz + 1:nc) .- 1])
#~     T = Toeplitz(cs[collect(nc - nlz:nc - 1)], cs[collect(reverse(nc - 2 * nlz + 1:nc - nlz))])
    as = T \ ks
    lambdas = roots(Polynomial(push!(reverse(as), -1.)))
    V = zeros(ComplexF64, nlz, nlz)
    for i in 1:nlz
        V[i, :] .= lambdas.^(i - 1)
    end

    ds = V \ ks ./ lambdas .^ (nc - nlz + 1)
    out::Array{ComplexF64, 1} = []
    for i in 1:nlz
        if abs(1 - real(ds[i])) < eps
            push!(out, time - 1im / lambdas[i])
        end
    end

    return out
end
