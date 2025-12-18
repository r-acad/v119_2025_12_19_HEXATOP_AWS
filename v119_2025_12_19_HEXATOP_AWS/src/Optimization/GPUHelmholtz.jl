# // # FILE: .\src\Optimization\GPUHelmholtz.jl

module GPUHelmholtz

using CUDA
using LinearAlgebra
using Printf
using ..Element
using ..Mesh

export HelmholtzWorkspace, setup_helmholtz_workspace, apply_gpu_filter!

mutable struct HelmholtzWorkspace{T}
    is_initialized::Bool
    radius::T
    
    # Grid Topology
    elements::CuVector{Int32} 
    Ae_base::CuMatrix{T}           
    inv_diag::CuVector{T}         
    
    # Solver Vectors
    r::CuVector{T}
    p::CuVector{T}
    z::CuVector{T}
    Ap::CuVector{T}
    x::CuVector{T} 
    b::CuVector{T} 
    
    # Coloring Cache
    color_counts::CuVector{Int32}
    color_offsets::CuVector{Int32}
    active_ids::CuVector{Int32} # Permutation map for coloring
    
    nNodes::Int
    nElem::Int
    
    HelmholtzWorkspace{T}() where T = new{T}(false, T(0))
end

const GLOBAL_HELMHOLTZ_CACHE = HelmholtzWorkspace{Float32}()

# -----------------------------------------------------------------------------
# COLORING HELPER
# -----------------------------------------------------------------------------
function get_element_color(e_idx_0based, nx, ny)
    slice = nx * ny
    iz = div(e_idx_0based, slice)
    rem = e_idx_0based % slice
    iy = div(rem, nx)
    ix = rem % nx
    return (ix % 2) + 2 * (iy % 2) + 4 * (iz % 2) + 1 
end

# -----------------------------------------------------------------------------
# KERNELS
# -----------------------------------------------------------------------------

function compute_rhs_kernel!(b, density, elements, val_scale, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        val = density[e] * val_scale
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            CUDA.atomic_add!(pointer(b, node), val)
        end
    end
    return nothing
end

# Optimized MatVec using Graph Coloring (No Atomics)
function matvec_helmholtz_kernel!(y, x, elements, Ae, active_ids, n_batch, batch_offset)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_batch
        actual_idx = idx + batch_offset
        e = active_ids[actual_idx]
        
        base_idx = (e - 1) * 8
        
        # Load x values
        x1 = x[elements[base_idx + 1]]; x2 = x[elements[base_idx + 2]]
        x3 = x[elements[base_idx + 3]]; x4 = x[elements[base_idx + 4]]
        x5 = x[elements[base_idx + 5]]; x6 = x[elements[base_idx + 6]]
        x7 = x[elements[base_idx + 7]]; x8 = x[elements[base_idx + 8]]
        
        @inbounds for r in 1:8
            val = Ae[r,1]*x1 + Ae[r,2]*x2 + Ae[r,3]*x3 + Ae[r,4]*x4 +
                  Ae[r,5]*x5 + Ae[r,6]*x6 + Ae[r,7]*x7 + Ae[r,8]*x8
            
            node = elements[base_idx + r]
            # No atomic_add needed due to coloring
            y[node] += val
        end
    end
    return nothing
end

function extract_solution_kernel!(filtered_density, x, elements, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        sum_val = 0.0f0
        @inbounds for i in 1:8
            sum_val += x[elements[base_idx + i]]
        end
        filtered_density[e] = sum_val / 8.0f0
    end
    return nothing
end

function compute_diagonal_kernel!(diag, elements, Ae_diag, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            val = Ae_diag[i]
            CUDA.atomic_add!(pointer(diag, node), val)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# SETUP & SOLVE
# -----------------------------------------------------------------------------

function setup_helmholtz_workspace(elements_cpu::Matrix{Int}, 
                                   dx::T, dy::T, dz::T, radius::T, 
                                   nx::Int, ny::Int) where T
    
    # We use a distinct cache for Blocked solves to allow resizing
    ws = HelmholtzWorkspace{T}() 
    nElem = size(elements_cpu, 1)
    nNodes = maximum(elements_cpu)
    
    # Matrix setup
    Ke, Me = Element.get_scalar_canonical_matrices(Float32(dx), Float32(dy), Float32(dz))
    Ae_cpu = Matrix{T}((radius^2) .* Ke .+ Me)
    
    elements_flat = vec(elements_cpu') 
    ws.elements = CuArray(Int32.(elements_flat))
    ws.Ae_base = CuArray(Ae_cpu)
    
    # Compute Diagonal (requires atomics, done once)
    diag_vec = CUDA.zeros(T, nNodes)
    Ae_diag_gpu = CuArray(diag(Ae_cpu))
    
    threads = 256
    blocks = cld(nElem, threads)
    @cuda threads=threads blocks=blocks compute_diagonal_kernel!(diag_vec, ws.elements, Ae_diag_gpu, nElem)
    ws.inv_diag = 1.0 ./ diag_vec
    
    # Coloring Setup for Fast MatVec
    colors = Vector{Int8}(undef, nElem)
    Base.Threads.@threads for e in 1:nElem
        colors[e] = Int8(get_element_color(e - 1, nx, ny))
    end
    
    perm = sortperm(colors)
    ws.active_ids = CuArray(Int32.(perm)) # ID mapping
    
    c_counts = zeros(Int32, 8)
    for c in colors; c_counts[c] += 1; end
    
    c_offsets = Vector{Int32}(undef, 8)
    curr = 0
    for i in 1:8; c_offsets[i] = Int32(curr); curr += c_counts[i]; end
    
    ws.color_counts = CuArray(c_counts)
    ws.color_offsets = CuArray(c_offsets)
    
    # Allocation
    ws.r  = CUDA.zeros(T, nNodes)
    ws.p  = CUDA.zeros(T, nNodes)
    ws.z  = CUDA.zeros(T, nNodes)
    ws.Ap = CUDA.zeros(T, nNodes)
    ws.x  = CUDA.zeros(T, nNodes)
    ws.b  = CUDA.zeros(T, nNodes)
    
    ws.nNodes = nNodes
    ws.nElem = nElem
    ws.radius = radius
    ws.is_initialized = true
    
    return ws
end

function solve_helmholtz_on_gpu(density_cpu::Vector{T}, ws::HelmholtzWorkspace{T}, 
                                dx, dy, dz, tol::T) where T
    
    density_gpu = CuArray(density_cpu) 
    filtered_gpu = CUDA.zeros(T, ws.nElem)
    
    # RHS (Requires atomics, but only once)
    fill!(ws.b, T(0.0))
    elem_vol = dx * dy * dz
    val_scale = T(elem_vol / 8.0)
    
    @cuda threads=256 blocks=cld(ws.nElem, 256) compute_rhs_kernel!(ws.b, density_gpu, ws.elements, val_scale, ws.nElem)
    
    norm_b = norm(ws.b)
    if norm_b == 0.0
        return density_cpu, 0.0, 0, 0.0
    end

    fill!(ws.x, T(0.0)) 
    ws.r .= ws.b
    ws.z .= ws.r .* ws.inv_diag 
    ws.p .= ws.z
    
    rho_old = dot(ws.r, ws.z)
    
    max_iter = 200 
    final_rel_res = 0.0
    final_iter = 0

    c_counts = Array(ws.color_counts)
    c_offsets = Array(ws.color_offsets)

    filter_start_time = time()

    for iter in 1:max_iter
        final_iter = iter
        
        fill!(ws.Ap, T(0.0))
        
        # Batched MatVec (No Atomics)
        for c in 1:8
            cnt = c_counts[c]
            off = c_offsets[c]
            if cnt > 0
                @cuda threads=256 blocks=cld(cnt, 256) matvec_helmholtz_kernel!(
                    ws.Ap, ws.p, ws.elements, ws.Ae_base, ws.active_ids, cnt, off
                )
            end
        end
        
        alpha = rho_old / dot(ws.p, ws.Ap)
        
        ws.x .+= alpha .* ws.p
        ws.r .-= alpha .* ws.Ap
        
        if iter % 10 == 0
            norm_r = norm(ws.r)
            final_rel_res = norm_r / norm_b
            if final_rel_res < tol
                break
            end
        end
        
        ws.z .= ws.r .* ws.inv_diag 
        
        rho_new = dot(ws.r, ws.z)
        beta = rho_new / rho_old
        ws.p .= ws.z .+ beta .* ws.p
        
        rho_old = rho_new
    end
    
    filter_time = time() - filter_start_time
    @cuda threads=256 blocks=cld(ws.nElem, 256) extract_solution_kernel!(filtered_gpu, ws.x, ws.elements, ws.nElem)
    
    return Array(filtered_gpu), filter_time, final_iter, final_rel_res
end

# -----------------------------------------------------------------------------
# BLOCKED DRIVER (WORKSPACE REUSE OPTIMIZATION)
# -----------------------------------------------------------------------------

function apply_blocked_gpu_filter!(density_full::Vector{T}, nElem_x, nElem_y, nElem_z, 
                                   dx, dy, dz, radius, tol::T) where T
    
    GC.gc(); CUDA.reclaim()
    
    bytes_per_elem_heuristic = 120 
    
    free_mem = CUDA.available_memory()
    safe_mem = Int(floor(free_mem * 0.80)) 
    safe_mem = max(safe_mem, 100 * 1024 * 1024) 
    
    max_elems_per_block = div(safe_mem, bytes_per_elem_heuristic)
    
    min_dim = min(dx, dy, dz)
    halo_cells = ceil(Int, (3.0 * radius) / min_dim)
    halo_cells = max(halo_cells, 2)
    
    cube_root = cbrt(max_elems_per_block)
    
    n_chunks_x = ceil(Int, nElem_x / cube_root)
    n_chunks_y = ceil(Int, nElem_y / cube_root)
    n_chunks_z = ceil(Int, nElem_z / cube_root)
    
    blk_nx = ceil(Int, nElem_x / n_chunks_x)
    blk_ny = ceil(Int, nElem_y / n_chunks_y)
    blk_nz = ceil(Int, nElem_z / n_chunks_z)

    filtered_full = zeros(T, length(density_full))
    
    total_time = 0.0
    total_iters = 0
    blocks_processed = 0
    
    # -------------------------------------------------------------------------
    # WORKSPACE REUSE STRATEGY
    # Allocate workspace ONCE for the largest possible block size
    # -------------------------------------------------------------------------
    max_loc_nx = blk_nx + 2*halo_cells
    max_loc_ny = blk_ny + 2*halo_cells
    max_loc_nz = blk_nz + 2*halo_cells
    
    # Create dummy mesh for allocation
    _, max_elems, _ = Mesh.generate_mesh(Int(max_loc_nx), Int(max_loc_ny), Int(max_loc_nz); dx=Float32(dx), dy=Float32(dy), dz=Float32(dz))
    
    # Pre-allocate Workspace
    ws = setup_helmholtz_workspace(max_elems, T(dx), T(dy), T(dz), T(radius), Int(max_loc_nx), Int(max_loc_ny))
    
    for cz in 1:n_chunks_z, cy in 1:n_chunks_y, cx in 1:n_chunks_x
        
        x_start = (cx - 1) * blk_nx + 1; x_end = min(cx * blk_nx, nElem_x)
        y_start = (cy - 1) * blk_ny + 1; y_end = min(cy * blk_ny, nElem_y)
        z_start = (cz - 1) * blk_nz + 1; z_end = min(cz * blk_nz, nElem_z)
        
        x_start_halo = max(1, x_start - halo_cells); x_end_halo = min(nElem_x, x_end + halo_cells)
        y_start_halo = max(1, y_start - halo_cells); y_end_halo = min(nElem_y, y_end + halo_cells)
        z_start_halo = max(1, z_start - halo_cells); z_end_halo = min(nElem_z, z_end + halo_cells)
        
        loc_nx = x_end_halo - x_start_halo + 1
        loc_ny = y_end_halo - y_start_halo + 1
        loc_nz = z_end_halo - z_start_halo + 1
        n_loc_elem = loc_nx * loc_ny * loc_nz
        
        # Check if current block fits in pre-allocated workspace
        if n_loc_elem > ws.nElem
             # Resize if needed (rare, only if edge blocks weirdly exceed estimate)
             _, elems_local, _ = Mesh.generate_mesh(loc_nx, loc_ny, loc_nz; dx=Float32(dx), dy=Float32(dy), dz=Float32(dz))
             ws = setup_helmholtz_workspace(elems_local, T(dx), T(dy), T(dz), T(radius), Int(loc_nx), Int(loc_ny))
        else
             # Update logical size without reallocation
             ws.nElem = n_loc_elem
             ws.nNodes = (loc_nx+1)*(loc_ny+1)*(loc_nz+1)
             
             # Need to regenerate element topology for specific block size
             _, elems_local, _ = Mesh.generate_mesh(loc_nx, loc_ny, loc_nz; dx=Float32(dx), dy=Float32(dy), dz=Float32(dz))
             elements_flat = vec(elems_local')
             
             # Reuse device memory
             copyto!(ws.elements, Int32.(elements_flat))
             
             # Re-compute diagonal and coloring for new topology
             fill!(ws.b, T(0.0)) # Use b as temp
             Ae_diag_gpu = CuArray(diag(Matrix(ws.Ae_base))) # Diagonal is small, low cost
             
             # Re-run diagonal kernel
             diag_vec = ws.b 
             fill!(diag_vec, T(0.0))
             @cuda threads=256 blocks=cld(n_loc_elem, 256) compute_diagonal_kernel!(diag_vec, ws.elements, Ae_diag_gpu, n_loc_elem)
             ws.inv_diag .= 1.0 ./ diag_vec
             
             # Re-run coloring (CPU side fast for small blocks)
             colors = Vector{Int8}(undef, n_loc_elem)
             for e in 1:n_loc_elem
                 colors[e] = Int8(get_element_color(e - 1, Int(loc_nx), Int(loc_ny)))
             end
             perm = sortperm(colors)
             copyto!(ws.active_ids, Int32.(perm))
             
             c_counts = zeros(Int32, 8)
             for c in colors; c_counts[c] += 1; end
             copyto!(ws.color_counts, c_counts)
             
             curr = 0; c_off = zeros(Int32, 8)
             for i in 1:8; c_off[i] = Int32(curr); curr += c_counts[i]; end
             copyto!(ws.color_offsets, c_off)
        end
        
        # Copy density chunk
        rho_local = zeros(T, n_loc_elem)
        loc_idx = 1
        for k in z_start_halo:z_end_halo
            for j in y_start_halo:y_end_halo
                global_start = (k-1)*(nElem_x*nElem_y) + (j-1)*nElem_x + x_start_halo
                rho_local[loc_idx : loc_idx+loc_nx-1] = density_full[global_start : global_start+loc_nx-1]
                loc_idx += loc_nx
            end
        end
        
        res_local, t_solve, iters, _ = solve_helmholtz_on_gpu(rho_local, ws, dx, dy, dz, tol)
        
        total_time += t_solve
        total_iters += iters
        blocks_processed += 1
        
        # Map back
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
                
                filtered_full[global_start_idx : global_start_idx+core_dim_x-1] = 
                    res_local[loc_start_idx : loc_start_idx+core_dim_x-1]
            end
        end
    end
    
    avg_iters = blocks_processed > 0 ? round(Int, total_iters/blocks_processed) : 0
    return filtered_full, total_time, avg_iters, 0.0
end


function apply_gpu_filter!(density_cpu::Vector{T}, nElem_x, nElem_y, nElem_z, dx, dy, dz, radius, elements_cpu, tol::T=1.0f-5) where T
    nElem = length(density_cpu)
    GC.gc(); CUDA.reclaim()
    
    free_mem = CUDA.available_memory()
    req_mem = nElem * 120 # Updated byte estimate for colored workspace
    
    try
        if req_mem < (free_mem * 0.9)
            ws = setup_helmholtz_workspace(elements_cpu, T(dx), T(dy), T(dz), T(radius), Int(nElem_x), Int(nElem_y))
            return solve_helmholtz_on_gpu(density_cpu, ws, dx, dy, dz, tol)
        else
            return apply_blocked_gpu_filter!(density_cpu, nElem_x, nElem_y, nElem_z, dx, dy, dz, radius, tol)
        end
    catch e
        println("\n" * "!"^60)
        println("  [GPU Filter] CRITICAL ERROR: $(e).")
        println("  [GPU Filter] Falling back to CPU Filter...")
        println("!"^60 * "\n")
        return Float32[], 0.0, 0, 0.0 
    end
end

end