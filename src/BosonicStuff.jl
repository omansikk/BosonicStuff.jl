module BosonicStuff
    include.(["Basis.jl", "Operators.jl", "Trajectories.jl", "Miscellaneous_quantities.jl", "Transform.jl"])
    
    # from Basis.jl
    export Basis_constant_N, Basis_global_max_N, Basis_local_max_N, Basis
    export dimension, length, find_vector, find_index, product_state, print_state
    
    # from Operators.jl
    export operator, diagonal_operator
    export number, numbers, annihilation, annihilations
    export ABH_Hamiltonian
    export number_disorder, anharmonicity_disorder
    
    # from Trajectories.jl
    export run_trajectories
    
    # from Miscellaneous_quantities.jl
    export inverse_participation_ratio, entropy
    export moments, loschmidt_zeros
    
    # from Transform.jl
    export transform
end
