# // # FILE: .\src\Optimization\TopOpt.jl

module TopologyOptimization 

using LinearAlgebra
using SparseArrays
using Printf  
using Statistics 
using SuiteSparse 
using CUDA
using Base.Threads
using ..Element
using ..Mesh
using ..GPUHelmholtz 
using ..Helpers

export update_density!, reset_filter_cache!

mutable struct FilterCache
    is_initialized::Bool
    radius::Float32
    K_filter::SuiteSparse.CHOLMOD.Factor{Float64} 
    FilterCache() = new(false, 0.0f0)
end

const GLOBAL_FILTER_CACHE = FilterCache()

function reset_filter_cache!()
    GLOBAL_FILTER_CACHE.is_initialized = false
end

function assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, R)
    nNodes = (nElem_x + 1) * (nElem_y + 1) * (nElem_z + 1)
    nElem = nElem_x * nElem_y * nElem_z
    Ke_local, Me_local = Element.get_scalar_canonical_matrices(dx, dy, dz)
    
    entries_per_elem = 64 
    total_entries = nElem * entries_per_elem
    
    I_vec = Vector{Int64}(undef, total_entries)
    J_vec = Vector{Int64}(undef, total_entries)
    V_vec = Vector{Float64}(undef, total_entries)
    
    nx, ny = nElem_x + 1, nElem_y + 1
    Re_local = (R^2) .* Ke_local .+ Me_local
    
    idx_counter = 0
    
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        n1 = i         + (j-1)*nx        + (k-1)*nx*ny
        n2 = (i+1) + (j-1)*nx        + (k-1)*nx*ny
        n3 = (i+1) + j*nx            + (k-1)*nx*ny
        n4 = i        + j*nx            + (k-1)*nx*ny
        n5 = i        + (j-1)*nx        + k*nx*ny
        n6 = (i+1) + (j-1)*nx        + k*nx*ny
        n7 = (i+1) + j*nx            + k*nx*ny
        n8 = i        + j*nx            + k*nx*ny
        nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        
        for r in 1:8
            row = nodes[r]
            for c in 1:8
                col = nodes[c]
                idx_counter += 1
                I_vec[idx_counter] = Int64(row)
                J_vec[idx_counter] = Int64(col)
                V_vec[idx_counter] = Float64(Re_local[r, c])
            end
        end
    end
    
    K_global = sparse(I_vec, J_vec, V_vec, nNodes, nNodes)
    
    n = size(K_global, 1)
    K_global = K_global + sparse(1:n, 1:n, fill(1e-9, n), n, n) 
    
    return cholesky(K_global)
end

function apply_helmholtz_filter_cpu(field_elem::Vector{Float32}, F_fact, nElem_x, nElem_y, nElem_z, dx, dy, dz)
    nx, ny = nElem_x + 1, nElem_y + 1
    nNodes = (nElem_x + 1) * (nElem_y + 1) * (nElem_z + 1)
    nElem = length(field_elem)
    
    elem_vol = dx * dy * dz
    nodal_weight = elem_vol / 8.0f0
    RHS = zeros(Float64, nNodes)
    
    idx_e = 1
    for k in 1:nElem_z, j in 1:nElem_y, i in 1:nElem_x
        val = Float64(field_elem[idx_e] * nodal_weight)
        n1 = i + (j-1)*nx + (k-1)*nx*ny
        n2 = n1 + 1
        n3 = n1 + nx + 1
        n4 = n1 + nx
        n5 = n1 + nx*ny
        n6 = n2 + nx*ny
        n7 = n3 + nx*ny
        n8 = n4 + nx*ny
        
        RHS[n1] += val; RHS[n2] += val; RHS[n3] += val; RHS[n4] += val;
        RHS[n5] += val; RHS[n6] += val; RHS[n7] += val; RHS[n8] += val;
        idx_e += 1
    end
    
    nodal_filtered = F_fact \ RHS
    
    filtered_elem = zeros(Float32, nElem)
    
    Threads.@threads for e in 1:nElem
        iz = div(e - 1, nElem_x * nElem_y) + 1
        rem_z = (e - 1) % (nElem_x * nElem_y)
        iy = div(rem_z, nElem_x) + 1
        ix = rem_z % nElem_x + 1
        
        n1 = ix + (iy-1)*nx + (iz-1)*nx*ny
        n2 = n1 + 1; n3 = n1 + nx + 1; n4 = n1 + nx;
        n5 = n1 + nx*ny; n6 = n2 + nx*ny; n7 = n3 + nx*ny; n8 = n4 + nx*ny;
        
        sum_nodes = nodal_filtered[n1] + nodal_filtered[n2] + nodal_filtered[n3] + nodal_filtered[n4] +
                    nodal_filtered[n5] + nodal_filtered[n6] + nodal_filtered[n7] + nodal_filtered[n8]
        
        filtered_elem[e] = Float32(sum_nodes / 8.0)
    end
    
    return filtered_elem
end

function apply_domain_decomposed_cpu_filter!(density_full::Vector{Float32}, 
                                             nElem_x, nElem_y, nElem_z, 
                                             dx, dy, dz, radius)
    
    target_elems_per_block = 50000
    cube_root = cbrt(target_elems_per_block)
    
    n_chunks_x = max(1, ceil(Int, nElem_x / cube_root))
    n_chunks_y = max(1, ceil(Int, nElem_y / cube_root))
    n_chunks_z = max(1, ceil(Int, nElem_z / cube_root))
    
    blk_nx = ceil(Int, nElem_x / n_chunks_x)
    blk_ny = ceil(Int, nElem_y / n_chunks_y)
    blk_nz = ceil(Int, nElem_z / n_chunks_z)
    
    min_dim = min(dx, dy, dz)
    halo_cells = max(2, ceil(Int, (3.0 * radius) / min_dim))
    
    filtered_full = zeros(Float32, length(density_full))
    
    blocks = []
    for cz in 1:n_chunks_z, cy in 1:n_chunks_y, cx in 1:n_chunks_x
        push!(blocks, (cx, cy, cz))
    end
    
    println("    [CPU Filter] Running Domain Decomposition: $(length(blocks)) blocks (Halo: $halo_cells)...")
    
    Threads.@threads for b_idx in 1:length(blocks)
        (cx, cy, cz) = blocks[b_idx]
        
        x_start = (cx - 1) * blk_nx + 1; x_end = min(cx * blk_nx, nElem_x)
        y_start = (cy - 1) * blk_ny + 1; y_end = min(cy * blk_ny, nElem_y)
        z_start = (cz - 1) * blk_nz + 1; z_end = min(cz * blk_nz, nElem_z)
        
        if x_start > x_end || y_start > y_end || z_start > z_end; continue; end

        x_start_halo = max(1, x_start - halo_cells); x_end_halo = min(nElem_x, x_end + halo_cells)
        y_start_halo = max(1, y_start - halo_cells); y_end_halo = min(nElem_y, y_end + halo_cells)
        z_start_halo = max(1, z_start - halo_cells); z_end_halo = min(nElem_z, z_end + halo_cells)
        
        loc_nx = x_end_halo - x_start_halo + 1
        loc_ny = y_end_halo - y_start_halo + 1
        loc_nz = z_end_halo - z_start_halo + 1
        n_loc_elem = loc_nx * loc_ny * loc_nz
        
        rho_local = zeros(Float32, n_loc_elem)
        loc_idx = 1
        for k in z_start_halo:z_end_halo
            for j in y_start_halo:y_end_halo
                global_start = (k-1)*(nElem_x*nElem_y) + (j-1)*nElem_x + x_start_halo
                copyto!(rho_local, loc_idx, density_full, global_start, loc_nx)
                loc_idx += loc_nx
            end
        end
        
        fact = assemble_helmholtz_system(loc_nx, loc_ny, loc_nz, dx, dy, dz, radius)
        res_local = apply_helmholtz_filter_cpu(rho_local, fact, loc_nx, loc_ny, loc_nz, dx, dy, dz)
        
        off_x = x_start - x_start_halo
        off_y = y_start - y_start_halo
        off_z = z_start - z_start_halo
        core_dim_x = x_end - x_start + 1
        
        for k in 1:(z_end - z_start + 1)
            loc_k = k + off_z
            for j in 1:(y_end - y_start + 1)
                loc_j = j + off_y
                
                loc_start_idx = (loc_k-1)*(loc_nx*loc_ny) + (loc_j-1)*loc_nx + (1 + off_x)
                
                global_k = z_start + k - 1
                global_j = y_start + j - 1
                global_start_idx = (global_k-1)*(nElem_x*nElem_y) + (global_j-1)*nElem_x + x_start
                
                copyto!(filtered_full, global_start_idx, res_local, loc_start_idx, core_dim_x)
            end
        end
    end
    
    return filtered_full
end

function update_density!(density::Vector{Float32}, 
                         l1_stress_norm_field::Vector{Float32}, 
                         protected_elements_mask::BitVector, 
                         E::Float32, 
                         l1_stress_allowable::Float32, 
                         iter::Int, 
                         number_of_iterations::Int, 
                         original_density::Vector{Float32}, 
                         min_density::Float32,  
                         max_density::Float32, 
                         config::Dict,
                         elements::Matrix{Int}, 
                         is_annealing::Bool=false) 

    nElem = length(density)
    
    # ------------------------------------------------------------------
    # SAFEGUARD: Check for NaNs
    # ------------------------------------------------------------------
    if any(isnan, l1_stress_norm_field)
        println("\n" * "\u001b[31m" * "!!!"^20 * "\u001b[0m")
        println("\u001b[31m" * ">>> [SAFEGUARD] CRITICAL: NaNs detected in stress field (Solver Diverged)." * "\u001b[0m")
        println("\u001b[31m" * ">>> [SAFEGUARD] Skipping topology update to prevent mesh corruption." * "\u001b[0m")
        println("\u001b[31m" * "!!!"^20 * "\n" * "\u001b[0m")
        return 0.0f0, 0.0f0, 0.0f0, 0.0, 0, 0.0
    end

    opt_params = config["optimization_parameters"]
    geom_params = config["geometry"]
    solver_params = config["solver_parameters"]
    
    nElem_x = Int(geom_params["nElem_x_computed"]) 
    nElem_y = Int(geom_params["nElem_y_computed"])
    nElem_z = Int(geom_params["nElem_z_computed"])
    dx = Float32(geom_params["dx_computed"])
    dy = Float32(geom_params["dy_computed"])
    dz = Float32(geom_params["dz_computed"])
    max_domain_dim = geom_params["max_domain_dim"]
    filter_tol = Float32(get(solver_params, "filter_tolerance", 1.0e-5))

    avg_element_size = (dx + dy + dz) / 3.0f0
    
    proposed_density_field = zeros(Float32, nElem)

    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            current_l1_stress = l1_stress_norm_field[e]
            val = (current_l1_stress / l1_stress_allowable) / E
            proposed_density_field[e] = max(val, min_density) 
        else
            proposed_density_field[e] = original_density[e]
        end
    end

    R_init_perc = Float32(get(opt_params, "filter_R_init_perc", 0.0f0))
    R_interm_perc = Float32(get(opt_params, "filter_R_interm_perc", 0.0f0))
    R_final_perc = Float32(get(opt_params, "filter_R_final_perc", 0.0f0))
    R_interm_iter_perc = Float32(get(opt_params, "filter_R_interm_iter_perc", 50.0f0)) / 100.0f0
    
    R_init_length = R_init_perc / 100.0f0 * max_domain_dim
    R_interm_length = R_interm_perc / 100.0f0 * max_domain_dim
    R_final_length = R_final_perc / 100.0f0 * max_domain_dim

    iter_interm = max(1, round(Int, R_interm_iter_perc * number_of_iterations))
    calc_iter = min(iter, number_of_iterations)
    
    R_length = 0.0f0

    if calc_iter <= iter_interm
        t = (iter_interm > 1) ? Float32(calc_iter - 1) / Float32(iter_interm - 1) : 0.0f0
        R_length = R_init_length * (1.0f0 - t) + R_interm_length * t
    else 
        t = (number_of_iterations > iter_interm) ? Float32(calc_iter - iter_interm) / Float32(number_of_iterations - iter_interm) : 0.0f0
        R_length = R_interm_length * (1.0f0 - t) + R_final_length * t
    end
    
    # -------------------------------------------------------------------------
    # MODIFIED: Use configurable minimum filter radius factor
    # -------------------------------------------------------------------------
    min_radius_factor = Float32(get(opt_params, "minimum_filter_radius_as_elements", 2.5))
    R_safe_min = min_radius_factor * avg_element_size
    
    if R_length < R_safe_min
        if iter % 10 == 0 
            println("    [Filter Guard] Radius < $(min_radius_factor)x Element Size. Auto-adjusting to ($R_safe_min).")
        end
        R_length = R_safe_min
    end
    
    R_effective = R_length / 2.5f0
    
    # -------------------------------------------------------------------------
    # Apply Filtering
    # -------------------------------------------------------------------------
    filtered_density_field = proposed_density_field
    filter_time = 0.0
    filter_iters = 0
    filter_res = 0.0
    
    if R_effective > 1e-4
        ran_gpu_successfully = false
        if CUDA.functional()
            filtered_gpu, t_gpu, it_gpu, res_gpu = GPUHelmholtz.apply_gpu_filter!(
                proposed_density_field, 
                nElem_x, nElem_y, nElem_z, 
                dx, dy, dz, R_effective, elements, filter_tol
            )
            
            if !isempty(filtered_gpu)
                filtered_density_field = filtered_gpu
                filter_time = t_gpu
                filter_iters = it_gpu
                filter_res = res_gpu
                ran_gpu_successfully = true
            end
        end

        if !ran_gpu_successfully
            t_start = time()
            if nElem < 100000
                if !GLOBAL_FILTER_CACHE.is_initialized || abs(GLOBAL_FILTER_CACHE.radius - R_effective) > 1e-5
                    println("    [CPU Filter] Building GLOBAL system for $nElem elements (R=$R_effective)...")
                    fact = assemble_helmholtz_system(nElem_x, nElem_y, nElem_z, dx, dy, dz, R_effective)
                    GLOBAL_FILTER_CACHE.K_filter = fact
                    GLOBAL_FILTER_CACHE.radius = R_effective
                    GLOBAL_FILTER_CACHE.is_initialized = true
                end
                
                filtered_density_field = apply_helmholtz_filter_cpu(
                    proposed_density_field, 
                    GLOBAL_FILTER_CACHE.K_filter, 
                    nElem_x, nElem_y, nElem_z, 
                    dx, dy, dz
                )
            else
                filtered_density_field = apply_domain_decomposed_cpu_filter!(
                    proposed_density_field, 
                    nElem_x, nElem_y, nElem_z, 
                    dx, dy, dz, R_effective
                )
            end
            filter_time = time() - t_start
            filter_iters = 1 
            filter_res = 0.0
        end
    end
    
    # -------------------------------------------------------------------------
    # Safety Check after Filtering
    # -------------------------------------------------------------------------
    if any(isnan, filtered_density_field)
        println("\u001b[33m" * ">>> [SAFEGUARD] Filter produced NaNs. Falling back to unfiltered density." * "\u001b[0m")
        filtered_density_field = proposed_density_field
    end
    
    # -------------------------------------------------------------------------
    # Apply Updates
    # -------------------------------------------------------------------------
    n_chunks = Threads.nthreads()
    chunk_size = cld(nElem, n_chunks)
    
    partial_stats = Vector{Tuple{Float32, Int}}(undef, n_chunks)
    
    @sync for (i, chunk_range) in enumerate(Iterators.partition(1:nElem, chunk_size))
        Threads.@spawn begin
            local_change = 0.0f0
            local_active = 0
            
            for e in chunk_range
                if !protected_elements_mask[e] 
                    old_val = density[e]
                    
                    # Update with filtered value
                    raw_new_val = max(filtered_density_field[e], min_density)
                    density[e] = raw_new_val
                    
                    effective_new_val = min(raw_new_val, max_density)
                    diff = abs(effective_new_val - old_val)
                    
                    if density[e] > min_density
                        local_active += 1
                    end
                    local_change += diff
                end
            end
            partial_stats[i] = (local_change, local_active)
        end
    end
    
    total_change = 0.0f0
    total_active = 0
    
    for i in 1:length(partial_stats)
        if isassigned(partial_stats, i)
            (c, a) = partial_stats[i]
            total_change += c
            total_active += a
        end
    end
    
    mean_change = (total_active > 0) ? (total_change / Float32(total_active)) : 0.0f0
    
    # -------------------------------------------------------------------------
    # Culling Logic
    # -------------------------------------------------------------------------
    current_threshold = 0.0f0
    final_threshold_val = Float32(get(opt_params, "final_density_threshold", 0.95))
    max_culling_ratio = Float32(get(opt_params, "max_culling_ratio", 0.1)) 
    
    if iter > number_of_iterations
        current_threshold = final_threshold_val
    else
        progress = clamp(Float32(iter) / Float32(number_of_iterations), 0f0, 1f0)
        current_threshold = final_threshold_val * progress
    end
    
    cull_candidates = Int[]
    active_count = 0
    
    for e in 1:nElem
        if !protected_elements_mask[e]
            if density[e] > max_density
                density[e] = max_density
            end
            
            if density[e] > min_density
                active_count += 1
                if density[e] < current_threshold
                    push!(cull_candidates, e)
                end
            end
        else
             if original_density[e] > min_density
                 active_count += 1
             end
        end
    end
    
    max_allowed_culls = floor(Int, active_count * max_culling_ratio)
    
    if length(cull_candidates) > max_allowed_culls
        sort!(cull_candidates, by = idx -> density[idx])
        for i in 1:max_allowed_culls
            idx = cull_candidates[i]
            density[idx] = min_density
        end
    else
        for idx in cull_candidates
            density[idx] = min_density
        end
    end

    update_method = get(opt_params, "density_update_method", "soft")
    
    if update_method == "hard"
        Threads.@threads for e in 1:nElem
            if !protected_elements_mask[e]
                if density[e] > min_density
                    density[e] = 1.0f0
                end
            end
        end
    end

    Threads.@threads for e in 1:nElem
        if protected_elements_mask[e]
            density[e] = original_density[e]
        end
    end
    
    return mean_change, R_length, current_threshold, filter_time, filter_iters, filter_res
end

end