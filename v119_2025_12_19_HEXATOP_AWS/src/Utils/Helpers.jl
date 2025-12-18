# FILE: .\src\Utils\Helpers.jl

module Helpers 

using CUDA 
using Printf

export expand_element_indices, nodes_from_location, parse_location_component 
export calculate_element_distribution, has_enough_gpu_memory, clear_gpu_memory, get_max_feasible_elements
export enforce_gpu_memory_safety

function expand_element_indices(elem_inds, dims) 
    nElem_x = dims[1] - 1 
    nElem_y = dims[2] - 1 
    nElem_z = dims[3] - 1 
    inds = Vector{Vector{Int}}() 
    for d in 1:3 
        if (typeof(elem_inds[d]) == String && elem_inds[d] == ":") 
            if d == 1 
                push!(inds, collect(1:nElem_x)) 
            elseif d == 2 
                push!(inds, collect(1:nElem_y)) 
            elseif d == 3 
                push!(inds, collect(1:nElem_z)) 
            end 
        else 
            push!(inds, [Int(elem_inds[d])]) 
        end 
    end 
    result = Int[] 
    for i in inds[1], j in inds[2], k in inds[3] 
        eidx = i + (j-1)*nElem_x + (k-1)*nElem_x*nElem_y 
        push!(result, eidx) 
    end 
    return result 
end 

function nodes_from_location(loc::Vector, dims) 
    nNodes_x, nNodes_y, nNodes_z = dims 
    ix = parse_location_component(loc[1], nNodes_x) 
    iy = parse_location_component(loc[2], nNodes_y) 
    iz = parse_location_component(loc[3], nNodes_z) 
    nodes = Int[] 
    for k in iz, j in iy, i in ix 
        node = i + (j-1)*nNodes_x + (k-1)*nNodes_x*nNodes_y 
        push!(nodes, node) 
    end 
    return nodes 
end 

function parse_location_component(val, nNodes::Int) 
    if val == ":" 
        return collect(1:nNodes) 
    elseif isa(val, String) && endswith(val, "%") 
        perc = parse(Float64, replace(val, "%"=>"")) / 100.0 
        idx = round(Int, 1 + perc*(nNodes-1)) 
        return [idx] 
    elseif isa(val, Number) 
        if 0.0 <= val <= 1.0 
            idx = round(Int, 1 + val*(nNodes-1)) 
            return [idx] 
        else 
            idx = clamp(round(Int, val), 1, nNodes) 
            return [idx] 
        end 
    else 
        error("Invalid location component: $val") 
    end 
end 

function clear_gpu_memory() 
    if !CUDA.functional() 
        return (0, 0) 
    end 
    GC.gc() 
    CUDA.reclaim() 

    final_free, total = CUDA.available_memory(), CUDA.total_memory() 
    return (final_free, total) 
end 

"""
    estimate_bytes_per_element(matrix_free::Bool)
"""
function estimate_bytes_per_element(matrix_free::Bool=true)
    if matrix_free
        return 220 
    else
        return 12000 
    end
end

"""
    get_max_feasible_elements(...)
"""
function get_max_feasible_elements(matrix_free::Bool=true; safety_factor::Float64=0.95, bytes_per_elem::Int=0)
    if !CUDA.functional() 
        return 5_000_000 
    end 
      
    free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory() 
    
    if total_mem > (70 * 1024^3) && safety_factor == 0.95
         safety_factor = 0.98
    end
    
    usable_mem = free_mem * safety_factor 
      
    bpe = (bytes_per_elem > 0) ? bytes_per_elem : estimate_bytes_per_element(matrix_free)
    
    max_elems = floor(Int, usable_mem / bpe) 
      
    return max_elems
end
 
function estimate_gpu_memory_required(nNodes, nElem, matrix_free::Bool=true) 
    return nElem * estimate_bytes_per_element(matrix_free)
end
 
function has_enough_gpu_memory(nNodes, nElem, matrix_free::Bool=true) 
    if !CUDA.functional() 
        return false 
    end 
    try 
        free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory() 
        required_mem = estimate_gpu_memory_required(nNodes, nElem, matrix_free) 
          
        utilization_limit = (total_mem > 70 * 1024^3) ? 0.98 : 0.90
        usable_mem = free_mem * utilization_limit 
          
        req_gb = required_mem / 1024^3
        avail_gb = usable_mem / 1024^3

        if required_mem > usable_mem
            @warn "GPU Memory Estimate:"
            @printf("   Required:  %.2f GB\n", req_gb)
            @printf("   Available: %.2f GB\n", avail_gb)
            println("   ⚠️ WARNING: Memory estimate exceeds safe limits.")
            return true 
        end
        return true 
    catch e 
        println("Error checking GPU memory: $e") 
        return true 
    end 
end 

"""
    calculate_element_distribution(length_x, length_y, length_z, target_elem_count)

Calculates the number of elements along each axis to achieve approximately `target_elem_count`
while maintaining cubic (or near-cubic) element aspect ratios based on the domain dimensions.
"""
function calculate_element_distribution(length_x, length_y, length_z, target_elem_count)
    total_volume = length_x * length_y * length_z
    k = cbrt(target_elem_count / total_volume)
    
    nElem_x = max(1, round(Int, k * length_x))
    nElem_y = max(1, round(Int, k * length_y))
    nElem_z = max(1, round(Int, k * length_z))

    dx = length_x / nElem_x
    dy = length_y / nElem_y
    dz = length_z / nElem_z
    
    actual_elem_count = nElem_x * nElem_y * nElem_z
    
    return nElem_x, nElem_y, nElem_z, Float32(dx), Float32(dy), Float32(dz), actual_elem_count
end

"""
    enforce_gpu_memory_safety(n_active_elem, n_nodes, use_double_precision, use_multigrid)

Performs a STRICT check of VRAM requirements against PHYSICAL available memory.
Throws an ErrorException if the simulation will crash or freeze the GPU.
"""
function enforce_gpu_memory_safety(n_active_elem::Int, n_nodes::Int, use_double_precision::Bool, use_multigrid::Bool)
    if !CUDA.functional(); return; end

    GC.gc()
    CUDA.reclaim()
    
    free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory()
    
    # --- PRECISE COST CALCULATION ---
    # Byte sizes
    float_size = use_double_precision ? 8 : 4
    int_size = 4 
    
    # 1. Connectivity (Int32) - 8 nodes per element
    mem_conn = n_active_elem * 8 * int_size
    
    # 2. Solver Vectors (Float) - DOF based
    n_dof_real = n_nodes * 3
    vec_count = 6 
    mem_vectors = n_dof_real * vec_count * float_size
    
    # 3. Multigrid Overhead 
    # UPDATED: MG is heavier than base CG (Coordinate arrays + vector fields).
    # Multiplier increased from 0.5 to 1.5 to match reality.
    mem_mg = use_multigrid ? (mem_vectors * 1.5) : 0
    
    # 4. Factors/Density/Misc
    mem_misc = n_active_elem * float_size * 2 
    
    total_required = mem_conn + mem_vectors + mem_mg + mem_misc
    
    # SAFETY BUFFER
    is_headless = (total_mem > 30 * 1024^3)
    safety_buffer = is_headless ? (1.0 * 1024^3) : (2.0 * 1024^3)
    
    if total_mem < 13 * 1024^3
        safety_buffer = 1.5 * 1024^3
    end

    available_for_compute = free_mem - safety_buffer
    
    req_gb = total_required / 1024^3
    avail_gb = free_mem / 1024^3
    safe_gb = available_for_compute / 1024^3
    
    if total_required > available_for_compute
        println("\n" * "\u001b[31m" * "!"^60 * "\u001b[0m")
        println("\u001b[31m" * ">>> [MEMORY GUARD] CRITICAL VRAM SHORTAGE DETECTED" * "\u001b[0m")
        println("\u001b[31m" * "!"^60 * "\u001b[0m")
        @printf("   Active Elements: %s\n", Base.format_bytes(n_active_elem))
        @printf("   Precision:       %s\n", use_double_precision ? "Float64 (Double)" : "Float32 (Single)")
        @printf("   Preconditioner:  %s\n", use_multigrid ? "Multigrid" : "Jacobi")
        println("-"^40)
        @printf("   Required Memory: %.2f GB\n", req_gb)
        @printf("   Physical Free:   %.2f GB\n", avail_gb)
        @printf("   Safe Limit:      %.2f GB (Buffer applied)\n", safe_gb)
        println("-"^40)
        println("   RESULT: The simulation was stopped to prevent system freeze/crash.")
        println("   ACTION: 1. Reduce 'target_active_elements'")
        println("           2. Switch to 'jacobi' preconditioner (saves ~30% RAM)")
        println("           3. Ensure 'gpu_profile' is NOT set to H100 (forces Float64)")
        println("\u001b[31m" * "!"^60 * "\u001b[0m\n")
        
        error("GPU Memory Guard: Insufficient VRAM for $(n_active_elem) elements.")
    else
        @printf("   [Memory Guard] %.2f GB required / %.2f GB available. Safe to proceed.\n", req_gb, avail_gb)
    end
end
 
end