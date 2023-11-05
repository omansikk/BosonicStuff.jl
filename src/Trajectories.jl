# This is very long, since it is essentially a collection of earlier ad-hoc stuff. I'm sure it's length could be cut by a lot, but it seems to be surprisingly difficult without a complete re-write.

# Bibliography:
# Saad, 'Analysis of Some Krylov Subspace Approximations to the Matrix Exponential Operator',  doi:10.1137/0729014
# Daley, 'Quantum trajectories and open many-body quantum systems', doi:10.1080/00018732.2014.933502


using Base.Threads, Distributed

mutable struct Work_arrays
    h::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
    w::Vector{ComplexF64}
end


"""
`init_work_arrays(dim::Int64, dim::Int64)`

Create arrays to use in the Krylov propagation algorithm in order to avoid unnecessary allocations.
"""
function init_work_arrays(dim::Int64, sdim::Int64)
    if sdim > 0
        return Work_arrays(zeros(ComplexF64, sdim, sdim), zeros(ComplexF64, dim, sdim), zeros(ComplexF64, dim))
    else
        return nothing
    end
end


"""
`set_timing_parameters!(saves, steps, dt, end_time)`

Set the timing parameters automatically depending on which ones haven't manually given.
"""
function set_timing_parameters(saves, steps, dt, end_time)
    save = 0
    if steps == 0 && dt == 0.
        steps = saves
        save = 1
        dt = end_time / steps
    elseif steps != 0 && dt == 0.
        dt = end_time / steps
        save = Int64(round(steps / saves))
    elseif dt != 0. && steps == 0
        steps = Int64(round(end_time / dt))
        save = Int64(floor(steps / saves))
        if steps * dt != end_time
            end_time = steps * dt
            println("\nWarning: end_time is not an integer multiple of dt. Automatically modified to end_time = " * string(end_time))
        end
    end

    return save, steps, dt, end_time
end


"""
`set_krylov_dimension(hamiltonian, state, dt; mindim = 2, maxdim = 30, maxerr = 1e-8)`

Computes a Krylov subspace dimension for a given `hamiltonian`, `state`, and `dt` with which the local error bound is at most `maxerr`. The actual error seems to usually be significantly smaller than the theoretical bound (see 'Analysis of Some Krylov Subspace Approximations to the Matrix Exponential Operator',  doi:10.1137/0729014).
"""
function set_krylov_dimension(hamiltonian, state, dt; mindim = 2, maxdim = length(state) - 1, maxerr = 1e-8)
    dim = length(state)
    h = zeros(ComplexF64, maxdim, maxdim)
    V = zeros(ComplexF64, dim, maxdim)
    krylov_dim = maxdim
    part = norm(Matrix(1.0im * dt .* hamiltonian))
    part *= exp(part)
    krylov_error = 0.
    V[:, 1] .= normalize(state)
    w = hamiltonian * state
    h[1, 1] = w' * state
    w .-= h[1, 1] * state
    for j in 2:maxdim
        beta = norm(w)
        if beta < 1e-8
            krylov_dim = j - 1
            break
        end

        V[:, j] .= w ./ beta
        w .= hamiltonian * @view V[:, j]
        h[j, j] = w' * @view V[:, j]
        @views w .-= (h[j, j] .* V[:, j] .+ beta .* V[:, j - 1])
        h[j - 1, j] = beta
        h[j, j - 1] = beta

        krylov_error = norm(Matrix(-1.0im * dt .* @view h[1:j, 1:j]))
        krylov_error = (krylov_error^j * exp(krylov_error) * norm(@view V[:, 1:j]) + part) / factorial(big(j))
        if j > mindim && abs(krylov_error) < maxerr
            krylov_dim = j
            break
        end
    end

    return krylov_dim
end


"""
`init_output_arrays(output_functions::Vector{Function}, initial_state::Vector, saves::Int64)`

Initialise arrays to store outputs.
"""
function init_output_array(output_functions::Vector{Function}, initial_state::Vector, saves::Int64)
    output_array::Vector{Vector} = []
    for i in 1:length(output_functions)
        output = output_functions[i](initial_state)
        push!(output_array, zeros(typeof(output), saves + 1))
        output_array[end][1] = output
    end

    return output_array
end


"""
`jump!(L, N, state, cs::Vector{SparseMatrixCSC})`

Randomly apply one of the jump operators `cs`.
"""
function jump!(L, N, state, cs)
    normalize!(state)
    channels = [real((cs[l] * state)' * cs[l] * state) for l in 1:L]
    channels ./= sum(channels)
    r = rand()
    for l in 1:L
        r -= channels[l]
        if r < 0.
            state .= normalize(cs[l] * state)
            break
        end
    end

    return nothing
end


"""
`function propagate!(hamiltonian::SparseMatrixCSC, state::Vector, dt::Float64, work_arrays::Work_arrays)`

Update the state using the Lanczos algorithm.
"""
function propagate!(hamiltonian::SparseMatrixCSC, state::Vector, dt::Float64, work_arrays::Work_arrays)
    beta = 0.
    work_arrays.V[:, 1] .= state
    work_arrays.w .= hamiltonian * state
    work_arrays.h[1, 1] = work_arrays.w' * state
    work_arrays.w .-= work_arrays.h[1, 1] * state
    for j in 2:size(work_arrays.h, 1)
        beta = norm(work_arrays.w)
        work_arrays.V[:, j] .= work_arrays.w ./ beta
        work_arrays.w .= hamiltonian * @view work_arrays.V[:, j]
        work_arrays.h[j, j] = work_arrays.w' * @view work_arrays.V[:, j]
        @views work_arrays.w .-= (work_arrays.h[j, j] .* work_arrays.V[:, j] .+ beta .* work_arrays.V[:, j - 1])
        work_arrays.h[j - 1, j] = beta
        work_arrays.h[j, j - 1] = beta
    end

    state .= work_arrays.V * @view exp(-(1.0im * dt) .* work_arrays.h)[1, :]

    return nothing
end


# single unitary trajectory with Krylov

function unitary_trajectory(hamiltonian::SparseMatrixCSC, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, work_arrays::Work_arrays, steps::Int64, save::Int64, dt::Float64)
    k = 2
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        propagate!(hamiltonian, state, dt, work_arrays)
        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end
            
            k += 1
        end
    end

    return output_array
end


# single unitary trajectory with Krylov for a time-dependent Hamiltonian

function unitary_trajectory(hamiltonian::SparseMatrixCSC, time_dependence::Function, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, work_arrays::Work_arrays, steps::Int64, save::Int64, dt::Float64)
    k = 2; t = 0.
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        propagate!(hamiltonian .+ time_dependence(t), state, dt, work_arrays)
        t += dt
        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end
            
            k += 1
        end
    end

    return output_array
end



# single unitary trajectory with full diagonalisation

function unitary_trajectory(exph::Matrix, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, steps::Int64, saves::Int64)
    k = 2
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        state .= exph * state
        if i % saves == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single unitary trajectory with full diagonalisation for a time-dependent Hamiltonian

function unitary_trajectory(H::Matrix, time_dependence::Function, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, steps::Int64, saves::Int64, dt::Float64)
    k = 2; t = 0.
    exph = zeros(ComplexF64, size(H))
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        exph .= exp(-1.0im * dt .* Matrix(H .+ time_dependence(t)))
        state .= exph * state
        t += dt
        if i % saves == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single non-unitary trajectory with dissipation or dephasing using the Krylov method

function nonunitary_trajectory(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, work_arrays::Work_arrays, cs::Vector, steps::Int64, save::Int64, L::Int64, N::Int64, dt::Float64)
    k = 2
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        propagate!(hamiltonian, state, dt, work_arrays)
        if real(state' * state) < rand()
            jump!(L, N, state, cs)
            if typeof(basis) != Basis_constant_N && state[1] == 1. # out of bosons
                break
            end
            
            krylov_dimension = set_krylov_dimension(hamiltonian, state, dt)
            work_arrays = init_work_arrays(length(initial_state), krylov_dimension)
        end

        normalize!(state)
        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single non-unitary trajectory with dissipation or dephasing and a time-dependent hamiltonian using the Krylov method

function nonunitary_trajectory(basis::Basis, hamiltonian::SparseMatrixCSC, time_dependence::Function, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, work_arrays::Work_arrays, cs::Vector, steps::Int64, save::Int64, L::Int64, N::Int64, dt::Float64)
    k = 2; t = 0.
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        propagate!(hamiltonian .+ time_dependence(t + 0.5 * dt), state, dt, work_arrays)
        t += dt
        if real(state' * state) < rand()
            jump!(L, N, state, cs)
            krylov_dimension = set_krylov_dimension(hamiltonian, state, dt)
            work_arrays = init_work_arrays(length(initial_state), krylov_dimension)
        end

        normalize!(state)
        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single non-unitary trajectory with dissipation or dephasing with full diagonalisation

function nonunitary_trajectory(basis::Basis, exph::Matrix, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, cs::Vector, steps::Int64, save::Int64, L::Int64, N::Int64)
    k = 2
    state::Vector{ComplexF64} = copy(initial_state)
    for i in 1:steps
        state .= exph * state
        if real(state' * state) < rand()
            jump!(L, N, state, cs)
            if typeof(basis) != Basis_constant_N && state[1] == 1. # out of bosons
                break
            end
        end

        normalize!(state)
        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single non-unitary trajectory with dissipation and dephasing using the Krylov method

function nonunitary_trajectory(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, work_arrays::Work_arrays, as::Vector, ns::Vector, dissipator::Vector, dephaser::Vector, steps::Int64, save::Int64, L::Int64, N::Int64, dt::Float64)
    k = 2
    state::Vector{ComplexF64} = copy(initial_state)
    test = copy(state)
    for i in 1:steps
        propagate!(hamiltonian, state, dt, work_arrays)
        normalize!(state)
        test .= dissipator .* state
        if real(test' * test) < rand()
            jump!(L, N, state, as)
            if typeof(basis) != Basis_constant_N && state[1] == 1. # out of bosons
                break
            end

            krylov_dimension = set_krylov_dimension(hamiltonian, state, dt)
            work_arrays = init_work_arrays(length(initial_state), krylov_dimension)
        end

        test .= dephaser .* state
        if real(test' * test) < rand()
            jump!(L, N, state, ns)
        end

        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single non-unitary trajectory with dissipation, dephasing, and a time-dependent hamiltonian using the Krylov method

function nonunitary_trajectory(basis::Basis, hamiltonian::SparseMatrixCSC, time_dependence::Function, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, work_arrays::Work_arrays, as::Vector, ns::Vector, dissipator::Vector, dephaser::Vector, steps::Int64, save::Int64, L::Int64, N::Int64, dt::Float64)
    k = 2; t = 0.
    state::Vector{ComplexF64} = copy(initial_state)
    test = copy(state)
    for i in 1:steps
        propagate!(hamiltonian .+ time_dependence(t + 0.5 * dt), state, dt, work_arrays)
        t += dt
        normalize!(state)
        test .= dissipator .* state
        if real(test' * test) < rand()
            jump!(L, N, state, as)
            if typeof(basis) != Basis_constant_N && state[1] == 1. # out of bosons
                break
            end

            krylov_dimension = set_krylov_dimension(hamiltonian, state, dt)
            work_arrays = init_work_arrays(length(initial_state), krylov_dimension)
        end

        test .= dephaser .* state
        if real(test' * test) < rand()
            jump!(L, N, state, ns)
        end

        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# single non-unitary trajectory with dissipation and dephasing with full diagonalisation

function nonunitary_trajectory(basis::Basis, exph::Matrix, initial_state::Vector{ComplexF64}, output_functions::Vector{Function}, output_array::Vector{Vector}, as::Vector, ns::Vector, dissipator::Vector, dephaser::Vector, steps::Int64, save::Int64, L::Int64, N::Int64)
    k = 2
    state::Vector{ComplexF64} = copy(initial_state)
    test = copy(state)
    for i in 1:steps
        state .= exph * state
        normalize!(state)
        test .= dissipator .* state
        if real(test' * test) < rand()
            jump!(L, N, state, as)
            if typeof(basis) != Basis_constant_N && state[1] == 1.
                break
            end
        end

        test .= dephaser .* state
        if real(test' * test) < rand()
            jump!(L, N, state, ns)
        end

        if i % save == 0
            for j in 1:length(output_functions)
                output_array[j][k] = output_functions[j](state)
            end

            k += 1
        end
    end

    return output_array
end


# unitary trajectories with or without disorder using the Krylov method

function run_unitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing, time_dependence = nothing)
    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            H = hamiltonian .+ disorder()
            if time_dependence != nothing
                unitary_trajectory(H, time_dependence, initial_state, output_functions, output_array, work_arrays, steps, save, dt)
            else
                unitary_trajectory(H, initial_state, output_functions, output_array, work_arrays, steps, save, dt)
            end
        end
    else
        if time_dependence != nothing
            unitary_trajectory(hamiltonian, time_dependence, initial_state, output_functions, output_array, work_arrays, steps, save, dt)
        else
            unitary_trajectory(hamiltonian, initial_state, output_functions, output_array, work_arrays, steps, save, dt)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end



# unitary trajectories with or without disorder with full diagonalisation

function run_unitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing, time_dependence = nothing)
    if time_dependence != nothing
        if disorder != nothing
            output_array .= @distributed (.+) for i in 1:trajectories
                unitary_trajectory(hamiltonian .+ Matrix(disorder()), time_dependence, initial_state, output_functions, output_array, steps, save, dt)
            end
        else
            unitary_trajectory(hamiltonian, time_dependence, initial_state, output_functions, output_array, steps, save, dt)
        end
    else
        if disorder != nothing
            output_array .= @distributed (.+) for i in 1:trajectories
                exph = exp(-1.0im * dt .* (hamiltonian .+ Matrix(disorder())))
                unitary_trajectory(exph, initial_state, output_functions, output_array, steps, save)
            end
        else
            exph = exp(-1.0im * dt .* hamiltonian)
            unitary_trajectory(exph, initial_state, output_functions, output_array, steps, save)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with random disorder in dissipation or dephasing using the Krylov method

function run_nonunitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, κ::Tuple{Float64, Float64}, cs::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    H0::SparseMatrixCSC{ComplexF64, Int64} = hamiltonian
    for l in 1:basis.L
        H0 .+= -0.5im * κ[1] .* cs[l]' * cs[l]
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[2] .* (2. .* rand(basis.L) .- 1.)
            H = copy(H0)
            cs2 = copy(cs)
            for l in 1:basis.L
                H .+= -0.5im * κ2[l] .* cs[l]' * cs[l]
                cs2[l] .*= sqrt(κ2[l])
            end

            nonunitary_trajectory(basis, H, initial_state, output_functions, output_array, work_arrays, cs2, steps, save, basis.L, basis.N, dt)
        end
    else
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[2] .* (2. .* rand(basis.L) .- 1.)
            H = copy(H0)
            cs2 = copy(cs)
            for l in 1:basis.L
                H .+= -0.5im * κ2[l] .* cs[l]' * cs[l]
                cs2[l] .*= sqrt(κ2[l])
            end

            nonunitary_trajectory(basis, initial_state, output_functions, output_array, work_arrays, cs2, steps, save, basis.L, basis.N, dt)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with random disorder in dissipation or dephasing with full diagonalisation

function run_nonunitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, κ::Tuple{Float64, Float64}, cs::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    H0::Matrix{ComplexF64} = hamiltonian
    for l in 1:basis.L
        H0 .+= -0.5im * κ[1] .* cs[l]' * cs[l]
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[2] .* (2. .* rand(basis.L) .- 1.)
            H = copy(H0)
            cs2 = copy(cs)
            for l in 1:basis.L
                H .+= -0.5im * κ2[l] .* cs[l]' * cs[l]
                cs2[l] .*= sqrt(κ2[l])
            end

            exph = exp(-1.0im * dt .* (H .+ Matrix(disorder())))

            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, cs2, steps, save, basis.L, basis.N)
        end
    else
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[2] .* (2. .* rand(basis.L) .- 1.)
            H = copy(H0)
            cs2 = copy(cs)
            for l in 1:basis.L
                H .+= -0.5im * κ2[l] .* cs[l]' * cs[l]
                cs2[l] .*= sqrt(κ2[l])
            end

            exph = exp(-1.0im * dt .* H)

            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, cs2, steps, save, basis.L, basis.N)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with a single fixed disorder pattern in dissipation or dephasing using the Krylov method

function run_nonunitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, κ::Vector{Float64}, cs::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    H0::SparseMatrixCSC{ComplexF64, Int64} = hamiltonian
    for l in 1:basis.L
        H0 .+= -0.5im * κ[l] .* cs[l]' * cs[l]
    end

    for l in 1:basis.L
        cs[l] .*= sqrt(κ[l])
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            H = H0 .+ disorder()
            nonunitary_trajectory(basis, H, initial_state, output_functions, output_array, work_arrays, cs, steps, save, basis.L, basis.N, dt)
        end
    else

        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, cs, steps, save, basis.L, basis.N, dt)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with a single fixed disorder pattern in dissipation or dephasing with full diagonalisation

function run_nonunitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, κ::Vector{Float64}, cs::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    H0::Matrix{ComplexF64} = hamiltonian
    for l in 1:basis.L
        H0 .+= -0.5im * κ[l] .* cs[l]' * cs[l]
    end
    for l in 1:basis.L
        cs[l] .*= sqrt(κ[l])
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            exph = exp(-1.0im * dt .* (H0 .+ Matrix(disorder())))
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, cs, steps, save, basis.L, basis.N,)
        end
    else
        exph = exp(-1.0im * dt .* H0)
        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, cs, steps, save, basis.L, basis.N)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with constant dissipation or dephasing using the Krylov method

function run_nonunitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, κ::Float64, cs::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    H0::SparseMatrixCSC{ComplexF64, Int64} = hamiltonian
    for l in 1:basis.L
        H0 .+= -0.5im * κ .* cs[l]' * cs[l]
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            H = H0 .+ disorder()
            nonunitary_trajectory(basis, H, initial_state, output_functions, output_array, work_arrays, cs, steps, save, basis.L, basis.N, dt)
        end
    else
        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, H0, initial_state, output_functions, output_array, work_arrays, cs, steps, save, basis.L, basis.N, dt)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with constant dissipation or dephasing with full diagonalisation

function run_nonunitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, κ::Float64, cs::Vector, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    H0::Matrix{ComplexF64} =  hamiltonian
    for l in 1:basis.L
        H0 .+= -0.5im * κ .* cs[l]' * cs[l]
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            exph = exp(-1.0im * dt .* (H0 .+ Matrix(disorder())))
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, cs, steps, save, basis.L, basis.N)
        end
    else
        exph = exp(-1.0im * dt .* H0)
        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, cs, steps, save, basis.L, basis.N)
        end
    end

    # output_array ./= trajectories?
    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with random disorder in dissipation and dephasing using the Krylov method

function run_nonunitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, γ::Tuple{Float64, Float64}, κ::Tuple{Float64, Float64}, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    as = annihilations(basis)
    ns = numbers(basis)
    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[1] .+ κ[2] .* (2. .* rand(basis.L) .- 1.)
            γ2 = γ[1] .+ γ[2] .* (2. .* rand(basis.L) .- 1.)
            dephaser = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(κ2 .* x.^2), element_type = ComplexF64))
            dissipator = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(γ2 .* x), element_type = ComplexF64))
            H = hamiltonian .+ disorder()
            nonunitary_trajectory(basis, H, initial_state, output_functions, output_array, work_arrays, as2, ns2, dissipator, dephaser, steps, save, basis.L, basis.N, dt)
        end
    else
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[1] .+ κ[2] .* (2. .* rand(basis.L) .- 1.)
            γ2 = γ[1] .+ γ[2] .* (2. .* rand(basis.L) .- 1.)
            dephaser = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(κ2 .* x.^2), element_type = ComplexF64))
            dissipator = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(γ2 .* x), element_type = ComplexF64))
            nonunitary_trajectory(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, as2, ns2, dissipator, dephaser, steps, save, basis.L, basis.N, dt)
        end
    end
    # output_array ./= trajectories?
    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with random disorder in dissipation and dephasing with full diagonalisation

function run_nonunitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, γ::Tuple{Float64, Float64}, κ::Tuple{Float64, Float64}, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    as = annihilations(basis)Krylov
    ns = numbers(basis)
    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[1] .+ κ[2] .* (2. .* rand(basis.L) .- 1.)
            γ2 = γ[1] .+ γ[2] .* (2. .* rand(basis.L) .- 1.)
            dephaser = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(κ2 .* x.^2), element_type = ComplexF64))
            dissipator = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(γ2 .* x), element_type = ComplexF64))
            exph = exp(-1.0im * dt .* (hamiltonian .+ Matrix(disorder())))
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, work_arrays, as2, ns2, dissipator, dephaser, steps, save, basis.L, basis.N)
        end
    else
        exph = exp(-1.0im * dt .* hamiltonian)
        output_array .= @distributed (.+) for i in 1:trajectories
            κ2 = κ[1] .+ κ[2] .* (2. .* rand(basis.L) .- 1.)
            γ2 = γ[1] .+ γ[2] .* (2. .* rand(basis.L) .- 1.)
            dephaser = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(κ2 .* x.^2), element_type = ComplexF64))
            dissipator = exp.(-0.5im * dt .* diagonal_operator(basis, x -> sum(γ2 .* x), element_type = ComplexF64))
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, work_arrays, as2, ns2, dissipator, dephaser, steps, save, basis.L, basis.N)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with a single fixed disorder pattern in dissipation and dephasing using the Krylov method

function run_nonunitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, γ::Vector{Float64}, κ::Vector{Float64}, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    dephaser = exp.(-dt .* diagonal_operator(basis, x -> sum(κ .* x.^2), element_type = ComplexF64))
    dissipator = exp.(-dt .* diagonal_operator(basis, x -> sum(γ .* x), element_type = ComplexF64))
    as = annihilations(basis)
    ns = numbers(basis)
    for l in 1:basis.L
        as[l] .*= γ[l]
        ns[l] .*= κ[l]
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            exph = exp(-1.0im * dt .* (hamiltonian .+ disorder()))
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N, dt)
        end
    else
        exph = exp(-1.0im * dt .* hamiltonian)
        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N, dt)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with a single fixed disorder pattern in dissipation and dephasing with full diagonalisation

function run_nonunitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, γ::Vector{Float64}, κ::Vector{Float64}, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    dephaser = exp.(-dt .* diagonal_operator(basis, x -> sum(κ .* x.^2), element_type = ComplexF64))
    dissipator = exp.(-dt .* diagonal_operator(basis, x -> sum(γ .* x), element_type = ComplexF64))
    as = annihilations(basis)
    ns = numbers(basis)
    for l in 1:basis.L
        as[l] .*= γ[l]
        ns[l] .*= κ[l]
    end

    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            exph = exp(-1.0im * dt .* (hamiltonian .+ disorder()))
            nonunitary_trajectory(basis, H, initial_state, output_functions, output_array, work_arrays, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N)
        end
    else
        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with constant dissipation and dephasing using the Krylov method

function run_nonunitary_trajectories(basis::Basis, hamiltonian::SparseMatrixCSC, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, work_arrays::Work_arrays, γ::Float64, κ::Float64, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    dephaser = exp.(-dt * κ .* diagonal_operator(basis, x -> sum(x.^2), element_type = ComplexF64))
    dissipator = exp.(-dt * γ .* diagonal_operator(basis, x -> sum(x), element_type = ComplexF64))
    as = annihilations(basis)
    ns = numbers(basis)
    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            H = hamiltonian .+ disorder()
            nonunitary_trajectory(basis, H, initial_state, output_functions, output_array, work_arrays, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N, dt)
        end
    else
        output_array .= @distributed (.+) for i in 1:trajectories
        nonunitary_trajectory(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N, dt)
        end
    end

    # output_array ./= trajectories?
    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


# non-unitary trajectories with constant dissipation and dephasing with full diagonalisation

function run_nonunitary_trajectories(basis::Basis, hamiltonian::Matrix, initial_state::Vector, output_functions::Vector{Function}, output_array::Vector, γ::Float64, κ::Float64, save::Int64, steps::Int64, trajectories::Int64, dt::Float64; disorder = nothing)
    dephaser = exp.(-dt * κ .* diagonal_operator(basis, x -> sum(x.^2), element_type = ComplexF64))
    dissipator = exp.(-dt * γ .* diagonal_operator(basis, x -> sum(x), element_type = ComplexF64))
    as = annihilations(basis)
    ns = numbers(basis)
    if disorder != nothing
        output_array .= @distributed (.+) for i in 1:trajectories
            exph = exp(-1.0im * dt .* (hamiltonian .+ Matrix(disorder())))
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N)
        end
    else
        exph = exp(-1.0im * dt .* hamiltonian)
        output_array .= @distributed (.+) for i in 1:trajectories
            nonunitary_trajectory(basis, exph, initial_state, output_functions, output_array, as, ns, dissipator, dephaser, steps, save, basis.L, basis.N)
        end
    end

    for i in 1:length(output_array)
        output_array[i] ./= trajectories
    end

    return output_array
end


"""
`run_trajectories(basis::Basis, hamiltonian, initial_state, output_functions::Vector{Function}, end_time; γ = 0, κ = 0, disorder = nothing, saves = 1000, steps = 0, dt = 0., trajectories = 1, krylov_dimension = 0)`

Run unitary or non-unitary trajectories.

Output functions should be a list of functions that take the current state as an input.

To include disorder in unitary parameters, define a function (passed as `disorder`) that takes no arguments and returns a sparse matrix representation of the disorder Hamiltonian.

If the Hamiltonian `hamiltonian` is a `SparseMatrixCSC`, the trajectories are run with the Krylov method. If it is a dense `Matrix`, the trajectories are run via full diagonalisation.

The types of the dissipation and dephasing rates γ, κ can be `Float64` for uniform rates, `Vector{Float64}` for constant disorder patterns, or `Tuple{Float64, Float64}`. If given as tuples, the first value is the mean rate, and the second is the half-width of a uniform distribution.

The parameters `steps`, `dt`, and `krylov_dimension`, if relevant, are determined automatically based on `saves` and `end_time` if not manually set. Actually returns output vectors of length `saves` + 1, where the +1 is the initial state. 
"""
function run_trajectories(basis::Basis, hamiltonian, initial_state::Vector, output_functions::Vector{Function}, end_time::Float64; κ = 0, γ = 0, disorder = nothing, time_dependence = nothing, saves::Int64 = 1000, steps::Int64 = 0, dt::Float64 = 0., trajectories::Int64 = 1, krylov_dimension::Int64 = 0)
    output_array = init_output_array(output_functions, initial_state, saves)
    save, steps, dt, end_time = set_timing_parameters(saves, steps, dt, end_time)
    if hamiltonian isa SparseMatrixCSC
        if krylov_dimension == 0
            krylov_dimension = set_krylov_dimension(hamiltonian, initial_state, dt)
        end

        work_arrays = init_work_arrays(length(initial_state), krylov_dimension)
        if γ == 0 && κ == 0

            return run_unitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, save, steps, trajectories, dt, disorder = disorder, time_dependence = time_dependence)

        elseif γ == 0 && κ != 0
            cs = numbers(basis)

            return run_nonunitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, κ, cs, save, steps, trajectories, dt, disorder = disorder)

        elseif γ != 0 && κ == 0
            cs = annihilations(basis)

            return run_nonunitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, γ, cs, save, steps, trajectories, dt, disorder = disorder)

        elseif γ != 0 && κ != 0

            return run_nonunitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, work_arrays, γ, κ, save, steps, trajectories, dt, disorder = disorder)

        end

    elseif hamiltonian isa Matrix
        if γ == 0 && κ == 0

            return run_unitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, save, steps, trajectories, dt, disorder = disorder, time_dependence = time_dependence)

        elseif γ == 0 && κ != 0
            cs = numbers(basis)

            return run_nonunitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, κ, cs, save, steps, trajectories, dt, disorder = disorder)

        elseif γ != 0 && κ == 0
            cs = annihilations(basis)

            return run_nonunitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, γ, cs, save, steps, trajectories, dt, disorder = disorder)


        elseif γ != 0 && κ != 0

            return run_nonunitary_trajectories(basis, hamiltonian, initial_state, output_functions, output_array, γ, κ, save, steps, trajectories, dt, disorder = disorder)

        end
    end
end

#to do: finish adding time-dependence to non-unitary trajectories
