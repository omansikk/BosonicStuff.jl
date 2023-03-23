using LinearAlgebra, Multisets, Combinatorics

function number_of_multiset_permutations(vector)
    m = Multiset(vector)
    s = unique(vector)
    n = factorial(length(vector))
    for i in s
        n /= factorial(m[i])
    end
    
    return n
end


function get_sites(input, output)
    js = Int64[]
    ks = Int64[]
    c = 1.
    for k in 1:length(input)
        if input[k] != 0
            c *= sqrt(factorial(input[k]))
            for l in 1:input[k]
                push!(js, k)
            end
        end

        if output[k] != 0
            c *= sqrt(factorial(output[k]))
            for l in 1:output[k]
                push!(ks, k)
            end
        end
    end
    
    return c, js, ks
end




"""
transform(basis::Basis; periodic = false)

Construct a matrix representation of transformation to the basis of the hopping term. 

NB this is essentially the old-timey Fourier transform and gets really slow with even moderate system sizes. Probably should be replaced by something like FFT.
"""
function transform(basis::Basis; periodic = false)
    dim = length(basis)
    T = zeros(dim, dim)
    if periodic == false
        Lp1 = 1. / (basis.L + 1)
        for i in 1:dim
            for j in i:dim
                c, js, ks = get_sites(basis[i], basis[j])
                c *= number_of_multiset_permutations(js)
                for output in multiset_permutations(ks, basis.N)
                    Tij = c
                    for l in 1:basis.N
                        Tij *= sin(pi * js[l] * output[l] * Lp1)
                    end

                    T[i, j] += Tij
                end
            end
        end

#~      n = (sqrt((L + 1) / 2) ^ N * factorial(N))
    else
        T = zeros(ComplexF64, dim, dim)
        Lp1 = 1. / (L)
        for i in 1:dim
            for j in i:dim
                c, js, ks = get_sites(basis[i], basis[j])
                c *= number_of_multiset_permutations(js)
                for output in multiset_permutations(ks, N)
                    Tij = c
                    for l in 1:basis.N
                        Tij *= exp(-2 * im * pi * js[l] * output[l] * Lp1)
                    end

                    T[i, j] += Tij
                end
            end
        end

    #~     n = (sqrt((L + 1) / 2) ^ N * factorial(N))
    end
        
    n = norm(T[1, :])
    T ./= n
    
    return Symmetric(T)
end


#~ function transform_periodic(L, N)
#~     input_vector = zeros(Int64, L)
#~     input_vector[1] = N
#~     dim = dimension(L, N)
#~     T = zeros(ComplexF64, dim, dim)
#~     Lp1 = 1. / (L)
#~     for i in 1:dim
#~         if i != 1
#~             next!(input_vector)
#~         end
    
#~         output_vector = copy(input_vector)
#~         for j in i:dim
#~             if j != i
#~                 next!(output_vector)
#~             end
            
#~             c, js, ks = get_sites(input_vector, output_vector)
#~             c *= number_of_multiset_permutations(js)
#~ #            println(i, " ", j)
#~             for output in multiset_permutations(ks, N)
#~                 Tij = c
#~                 for l in 1:N
#~                     Tij *= exp(-2 * im * pi * js[l] * output[l] * Lp1)
#~                 end

#~                 T[i, j] += Tij
#~ #                println(js, " ", output, " ", Tij)
#~             end
#~         end
#~     end

#~     n = norm(T[1, :])
#~     n = (sqrt((L + 1) / 2) ^ N * factorial(N))
#~     T ./= n
#~     return Hermitian(T)
#~     return Symmetric(T)
#~ end
