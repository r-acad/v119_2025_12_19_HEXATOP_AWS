# // # FILE: .\src\Mesh\MeshRefiner.jl

module MeshRefiner

using LinearAlgebra
using Printf
using Base.Threads
using ..Mesh
using ..Helpers

export refine_mesh_and_fields

function estimate_element_memory_cost_bytes(hard_element_limit::Int)
    if hard_element_limit > 500_000_000 
        return 180 
    else
        return 250 
    end
end

function get_safe_element_limit(hard_element_limit::Int)
    free_ram = Float64(Sys.free_memory())
    is_h_series = hard_element_limit > 500_000_000
    
    safety_buffer = if is_h_series
        max(free_ram * 0.05, 4.0 * 1024^3) 
    else
        max(free_ram * 0.15, 6.0 * 1024^3) 
    end
        
    usable_ram = free_ram - safety_buffer

    if usable_ram <= 0
        @warn "[MeshRefiner] System RAM is critically low. Limiting mesh severely."
        return 1_000_000 
    end

    bytes_per_elem = estimate_element_memory_cost_bytes(hard_element_limit)
    max_elements = floor(Int, usable_ram / bytes_per_elem)
    
    return max_elements
end

@inline function interpolate_trilinear_projected(
    v000::Float32, v100::Float32, v010::Float32, v110::Float32,
    v001::Float32, v101::Float32, v011::Float32, v111::Float32,
    tx::Float32, ty::Float32, tz::Float32, 
    use_projection::Bool
)
    ctx = clamp(tx, 0.0f0, 1.0f0)
    cty = clamp(ty, 0.0f0, 1.0f0)
    ctz = clamp(tz, 0.0f0, 1.0f0)

    c00 = v000 * (1 - ctx) + v100 * ctx
    c10 = v010 * (1 - ctx) + v110 * ctx
    c01 = v001 * (1 - ctx) + v101 * ctx
    c11 = v011 * (1 - ctx) + v111 * ctx

    c0 = c00 * (1 - cty) + c10 * cty
    c1 = c01 * (1 - cty) + c11 * cty

    val = c0 * (1 - ctz) + c1 * ctz

    if use_projection
        if val > 0.4f0
            return 1.0f0
        elseif val < 0.2f0
            return 0.0001f0
        else
            return val
        end
    end

    return val
end

function refine_mesh_and_fields(nodes::Matrix{Float32}, 
                                elements::Matrix{Int}, 
                                density::Vector{Float32}, 
                                alpha_field::Vector{Float32}, 
                                current_dims::Tuple{Int, Int, Int},
                                target_active_count::Int,
                                domain_bounds::NamedTuple;
                                max_growth_rate::Float64=1.2,
                                hard_element_limit::Int=800_000_000,
                                update_method::String="hard") 

    C_RESET = "\u001b[0m"
    C_BOLD = "\u001b[1m"
    C_CYAN = "\u001b[36m"
    C_GREEN = "\u001b[32m"
    C_YELLOW = "\u001b[33m"
    C_RED = "\u001b[31m"

    println("\n" * C_CYAN * "="^60 * C_RESET)
    println(C_CYAN * C_BOLD * ">>> [MESH REFINER] Evaluating Refinement Strategy" * C_RESET)

    n_total_old = length(density)
    n_active_old = count(d -> d > 0.001f0, density) 
    active_ratio = max(0.0001, n_active_old / n_total_old) 
    
    println("    Current Total Elements:  $(n_total_old)")
    println("    Current Active Elements: $(n_active_old) ($(round(active_ratio*100, digits=2))%)")
    println("    Target Active Limit:     $(target_active_count)")

    ideal_total_from_target = round(Int, (target_active_count / active_ratio) * 0.90)
    rate_limit_elements = round(Int, n_total_old * max_growth_rate)
    ram_limit_elements = get_safe_element_limit(hard_element_limit)

    final_new_total = n_total_old

    if n_active_old >= target_active_count
        println(C_GREEN * "    [OK] Target active count reached. Maintaining mesh resolution." * C_RESET)
        final_new_total = n_total_old 
    else
        limits = [
            ("Target Active Limit", ideal_total_from_target),
            ("Growth Rate Limit", rate_limit_elements),
            ("RAM Limit", ram_limit_elements),
            ("Config Hard Limit", hard_element_limit)
        ]
        
        sort!(limits, by = x -> x[2])
        
        limiting_factor_name = limits[1][1]
        final_new_total = limits[1][2]
        
        println("    Constraint Analysis:")
        for (name, val) in limits
            col = (name == limiting_factor_name) ? C_YELLOW : C_RESET
            println("      - $col$name: $(Base.format_bytes(val * 200)) approx ($val elems)$C_RESET")
        end
        println("    LIMIT APPLIED: $C_YELLOW$limiting_factor_name$C_RESET")
    end

    if final_new_total > hard_element_limit
        final_new_total = hard_element_limit
    end

    if final_new_total < (n_total_old * 1.05)
        println(C_YELLOW * "    [SKIP] Calculated growth too small (< 5%). Skipping." * C_RESET)
        println(C_CYAN * "="^60 * "\n" * C_RESET)
        return nodes, elements, density, alpha_field, current_dims
    end

    println(C_GREEN * C_BOLD * "    >>> EXECUTING REFINEMENT TO: $final_new_total elements" * C_RESET)

    len_x, len_y, len_z = domain_bounds.len_x, domain_bounds.len_y, domain_bounds.len_z
    new_nx, new_ny, new_nz, new_dx, new_dy, new_dz, actual_count = 
        Helpers.calculate_element_distribution(len_x, len_y, len_z, final_new_total)
        
    println("      > Grid: $(new_nx)x$(new_ny)x$(new_nz) = $actual_count")
    println("      > Res:  $(new_dx) x $(new_dy) x $(new_dz)")

    if actual_count > get_safe_element_limit(hard_element_limit)
        println(C_RED * "    [!!!] CRITICAL WARNING: Memory race condition detected. Aborting." * C_RESET)
        return nodes, elements, density, alpha_field, current_dims
    end

    new_nodes, new_elements, new_dims = Mesh.generate_mesh(
        new_nx, new_ny, new_nz; 
        dx=new_dx, dy=new_dy, dz=new_dz
    )
    
    min_pt = domain_bounds.min_pt
    new_nodes[:, 1] .+= min_pt[1]
    new_nodes[:, 2] .+= min_pt[2]
    new_nodes[:, 3] .+= min_pt[3]
    
    println("      > Mapping density and thermal fields (Robust Interpolation)...")
    n_new_total = size(new_elements, 1)
    new_density = zeros(Float32, n_new_total)
    new_alpha    = zeros(Float32, n_new_total) 
    
    # Cast to Int64 to prevent intermediate overflow during indexing of large meshes
    old_nx = Int64(current_dims[1] - 1)
    old_ny = Int64(current_dims[2] - 1)
    old_nz = Int64(current_dims[3] - 1)
    
    new_nx_64 = Int64(new_nx)
    new_ny_64 = Int64(new_ny)
    
    use_projection = (update_method == "hard")

    Threads.@threads for e_new in 1:n_new_total
        # Calculate 3D index of new element
        e_new_idx = Int64(e_new)
        iz = div(e_new_idx - 1, new_nx_64 * new_ny_64) + 1
        rem_z = (e_new_idx - 1) % (new_nx_64 * new_ny_64)
        iy = div(rem_z, new_nx_64) + 1
        ix = rem_z % new_nx_64 + 1
        
        # Centroid of new element
        cx = (Float32(ix) - 0.5f0) * new_dx
        cy = (Float32(iy) - 0.5f0) * new_dy
        cz = (Float32(iz) - 0.5f0) * new_dz
        
        # Map to Old Grid "Continuous Index" space
        u = clamp((cx / len_x) * Float32(old_nx), 0.5f0, Float32(old_nx) - 0.5f0)
        v = clamp((cy / len_y) * Float32(old_ny), 0.5f0, Float32(old_ny) - 0.5f0)
        w = clamp((cz / len_z) * Float32(old_nz), 0.5f0, Float32(old_nz) - 0.5f0)

        # Trilinear base indices
        u0 = floor(Int64, u - 0.5f0)
        v0 = floor(Int64, v - 0.5f0)
        w0 = floor(Int64, w - 0.5f0)

        tx = (u - 0.5f0) - Float32(u0)
        ty = (v - 0.5f0) - Float32(v0)
        tz = (w - 0.5f0) - Float32(w0)

        # Clamped neighbors (1-based)
        i0 = clamp(u0 + 1, 1, old_nx); i1 = clamp(u0 + 2, 1, old_nx)
        j0 = clamp(v0 + 1, 1, old_ny); j1 = clamp(v0 + 2, 1, old_ny)
        k0 = clamp(w0 + 1, 1, old_nz); k1 = clamp(w0 + 2, 1, old_nz)

        # Robust Indexing (Int64 arithmetic)
        # i + (j-1)*nx + (k-1)*nx*ny
        slice_size = old_nx * old_ny
        
        idx000 = i0 + (j0-1)*old_nx + (k0-1)*slice_size
        idx100 = i1 + (j0-1)*old_nx + (k0-1)*slice_size
        idx010 = i0 + (j1-1)*old_nx + (k0-1)*slice_size
        idx110 = i1 + (j1-1)*old_nx + (k0-1)*slice_size
        idx001 = i0 + (j0-1)*old_nx + (k1-1)*slice_size
        idx101 = i1 + (j0-1)*old_nx + (k1-1)*slice_size
        idx011 = i0 + (j1-1)*old_nx + (k1-1)*slice_size
        idx111 = i1 + (j1-1)*old_nx + (k1-1)*slice_size

        new_density[e_new] = interpolate_trilinear_projected(
            density[idx000], density[idx100], density[idx010], density[idx110],
            density[idx001], density[idx101], density[idx011], density[idx111],
            tx, ty, tz, use_projection
        )

        new_alpha[e_new] = alpha_field[idx000] 
    end
    
    println(C_GREEN * "    [DONE] Refinement Complete." * C_RESET)
    println(C_CYAN * "="^60 * "\n" * C_RESET)

    return new_nodes, new_elements, new_density, new_alpha, new_dims
end

end