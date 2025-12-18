# // # FILE: .\src\Solvers\GPUGeometricMultigrid.jl

module GPUGeometricMultigrid

using CUDA
using LinearAlgebra
using SparseArrays
using Printf
using ..Element
using ..Mesh 

export MGWorkspace, setup_multigrid, apply_mg_vcycle!

mutable struct MGWorkspace
    is_initialized::Bool
    
    nc_x::Int; nc_y::Int; nc_z::Int
    
    r_coarse::Any
    x_coarse::Any
    diag_coarse::Any
    
    density_coarse::Any
    conn_coarse::Any
    
    fine_node_coords_normalized::Any 
    
    dx_c::Float32; dy_c::Float32; dz_c::Float32
    
    # Temp Full Vectors for Mapping
    r_fine_full::Any
    z_fine_full::Any
    
    MGWorkspace() = new(false, 0,0,0, nothing, nothing, nothing, nothing, nothing, nothing, 1.0, 1.0, 1.0, nothing, nothing)
end

const GLOBAL_MG_CACHE = MGWorkspace()

# --- LOCAL KERNELS ---

function compute_diagonal_atomic_kernel!(diag_vec, conn, Ke_diag, factors, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        factor = factors[e]
        for i in 1:8
            node = conn[base_idx + i]
            k_val = Ke_diag[(i-1)*3 + 1] * factor
            CUDA.atomic_add!(pointer(diag_vec, (node - 1) * 3 + 1), k_val)
            CUDA.atomic_add!(pointer(diag_vec, (node - 1) * 3 + 2), k_val)
            CUDA.atomic_add!(pointer(diag_vec, (node - 1) * 3 + 3), k_val)
        end
    end
    return nothing
end

function restrict_residual_kernel!(r_c, r_f, node_coords, nNodes_fine, nx_c, ny_c, nz_c)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nNodes_fine
        x = node_coords[idx, 1]; y = node_coords[idx, 2]; z = node_coords[idx, 3]
        
        i0 = floor(Int, x); i1 = i0 + 1
        j0 = floor(Int, y); j1 = j0 + 1
        k0 = floor(Int, z); k1 = k0 + 1
        
        tx = x - i0; ty = y - j0; tz = z - k0
        
        nx_nodes = nx_c + 1; ny_nodes = ny_c + 1; slice = nx_nodes * ny_nodes
        base_idx = 3 * (idx - 1)
        rf_x = r_f[base_idx + 1]; rf_y = r_f[base_idx + 2]; rf_z = r_f[base_idx + 3]
        
        # (0,0,0)
        w = (1-tx)*(1-ty)*(1-tz)
        if i0 >= 0 && j0 >= 0 && k0 >= 0 && i0 <= nx_c && j0 <= ny_c && k0 <= nz_c
            c_idx = (i0 + 1 + j0 * nx_nodes + k0 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (1,0,0)
        w = (tx)*(1-ty)*(1-tz)
        if i1 >= 0 && j0 >= 0 && k0 >= 0 && i1 <= nx_c && j0 <= ny_c && k0 <= nz_c
            c_idx = (i1 + 1 + j0 * nx_nodes + k0 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (0,1,0)
        w = (1-tx)*(ty)*(1-tz)
        if i0 >= 0 && j1 >= 0 && k0 >= 0 && i0 <= nx_c && j1 <= ny_c && k0 <= nz_c
            c_idx = (i0 + 1 + j1 * nx_nodes + k0 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (1,1,0)
        w = (tx)*(ty)*(1-tz)
        if i1 >= 0 && j1 >= 0 && k0 >= 0 && i1 <= nx_c && j1 <= ny_c && k0 <= nz_c
            c_idx = (i1 + 1 + j1 * nx_nodes + k0 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (0,0,1)
        w = (1-tx)*(1-ty)*(tz)
        if i0 >= 0 && j0 >= 0 && k1 >= 0 && i0 <= nx_c && j0 <= ny_c && k1 <= nz_c
            c_idx = (i0 + 1 + j0 * nx_nodes + k1 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (1,0,1)
        w = (tx)*(1-ty)*(tz)
        if i1 >= 0 && j0 >= 0 && k1 >= 0 && i1 <= nx_c && j0 <= ny_c && k1 <= nz_c
            c_idx = (i1 + 1 + j0 * nx_nodes + k1 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (0,1,1)
        w = (1-tx)*(ty)*(tz)
        if i0 >= 0 && j1 >= 0 && k1 >= 0 && i0 <= nx_c && j1 <= ny_c && k1 <= nz_c
            c_idx = (i0 + 1 + j1 * nx_nodes + k1 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
        # (1,1,1)
        w = (tx)*(ty)*(tz)
        if i1 >= 0 && j1 >= 0 && k1 >= 0 && i1 <= nx_c && j1 <= ny_c && k1 <= nz_c
            c_idx = (i1 + 1 + j1 * nx_nodes + k1 * slice - 1) * 3
            CUDA.atomic_add!(pointer(r_c, c_idx + 1), rf_x * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 2), rf_y * w)
            CUDA.atomic_add!(pointer(r_c, c_idx + 3), rf_z * w)
        end
    end
    return nothing
end

function prolongate_correction_kernel!(x_f, x_c, node_coords, nNodes_fine, nx_c, ny_c, nz_c)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nNodes_fine
        x = node_coords[idx, 1]; y = node_coords[idx, 2]; z = node_coords[idx, 3]
        
        i0 = floor(Int, x); i1 = i0 + 1
        j0 = floor(Int, y); j1 = j0 + 1
        k0 = floor(Int, z); k1 = k0 + 1
        
        tx = x - i0; ty = y - j0; tz = z - k0
        
        nx_nodes = nx_c + 1; ny_nodes = ny_c + 1; slice = nx_nodes * ny_nodes
        val_x, val_y, val_z = 0.0f0, 0.0f0, 0.0f0
        
        # (0,0,0)
        w = (1-tx)*(1-ty)*(1-tz)
        if i0 >= 0 && j0 >= 0 && k0 >= 0 && i0 <= nx_c && j0 <= ny_c && k0 <= nz_c
            c_base = (i0 + 1 + j0 * nx_nodes + k0 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (1,0,0)
        w = (tx)*(1-ty)*(1-tz)
        if i1 >= 0 && j0 >= 0 && k0 >= 0 && i1 <= nx_c && j0 <= ny_c && k0 <= nz_c
            c_base = (i1 + 1 + j0 * nx_nodes + k0 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (0,1,0)
        w = (1-tx)*(ty)*(1-tz)
        if i0 >= 0 && j1 >= 0 && k0 >= 0 && i0 <= nx_c && j1 <= ny_c && k0 <= nz_c
            c_base = (i0 + 1 + j1 * nx_nodes + k0 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (1,1,0)
        w = (tx)*(ty)*(1-tz)
        if i1 >= 0 && j1 >= 0 && k0 >= 0 && i1 <= nx_c && j1 <= ny_c && k0 <= nz_c
            c_base = (i1 + 1 + j1 * nx_nodes + k0 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (0,0,1)
        w = (1-tx)*(1-ty)*(tz)
        if i0 >= 0 && j0 >= 0 && k1 >= 0 && i0 <= nx_c && j0 <= ny_c && k1 <= nz_c
            c_base = (i0 + 1 + j0 * nx_nodes + k1 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (1,0,1)
        w = (tx)*(1-ty)*(tz)
        if i1 >= 0 && j0 >= 0 && k1 >= 0 && i1 <= nx_c && j0 <= ny_c && k1 <= nz_c
            c_base = (i1 + 1 + j0 * nx_nodes + k1 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (0,1,1)
        w = (1-tx)*(ty)*(tz)
        if i0 >= 0 && j1 >= 0 && k1 >= 0 && i0 <= nx_c && j1 <= ny_c && k1 <= nz_c
            c_base = (i0 + 1 + j1 * nx_nodes + k1 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        # (1,1,1)
        w = (tx)*(ty)*(tz)
        if i1 >= 0 && j1 >= 0 && k1 >= 0 && i1 <= nx_c && j1 <= ny_c && k1 <= nz_c
            c_base = (i1 + 1 + j1 * nx_nodes + k1 * slice - 1) * 3
            val_x += x_c[c_base + 1] * w; val_y += x_c[c_base + 2] * w; val_z += x_c[c_base + 3] * w
        end
        
        base_idx = 3 * (idx - 1)
        x_f[base_idx + 1] += val_x
        x_f[base_idx + 2] += val_y
        x_f[base_idx + 3] += val_z
    end
    return nothing
end

function normalize_coords_kernel!(out, pts, dx, dy, dz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(pts, 1)
        out[i, 1] = pts[i, 1] / dx
        out[i, 2] = pts[i, 2] / dy
        out[i, 3] = pts[i, 3] / dz
    end
    return nothing
end

function map_density_kernel!(rho_c, coords, nx, ny, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(coords, 1)
        ix = floor(Int, coords[i, 1])
        iy = floor(Int, coords[i, 2])
        iz = floor(Int, coords[i, 3])
        if ix >= 0 && iy >= 0 && iz >= 0 && ix < nx && iy < ny && iz < nz
            c_elem_idx = ix + 1 + iy * nx + iz * nx * ny
            rho_c[c_elem_idx] = 1.0f0 
        end
    end
    return nothing
end

function expand_kernel!(x_full, x_free, map, n_free)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_full[map[idx]] = x_free[idx]
    end
    return nothing
end

function contract_add_kernel!(y_free, y_full, map, n_free)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds y_free[idx] += y_full[map[idx]]
    end
    return nothing
end

# --- SETUP ---

function setup_multigrid(fine_nodes, fine_density, config)
    ws = GLOBAL_MG_CACHE
    geom = config["geometry"]
    dx_f = Float32(get(geom, "dx_computed", 1.0)); dy_f = Float32(get(geom, "dy_computed", 1.0)); dz_f = Float32(get(geom, "dz_computed", 1.0))
    nx_f = Int(get(geom, "nElem_x_computed", 0)); ny_f = Int(get(geom, "nElem_y_computed", 0)); nz_f = Int(get(geom, "nElem_z_computed", 0))
    
    nx_c = max(1, div(nx_f, 2)); ny_c = max(1, div(ny_f, 2)); nz_c = max(1, div(nz_f, 2))
    ws.dx_c = dx_f * 2.0; ws.dy_c = dy_f * 2.0; ws.dz_c = dz_f * 2.0
    ws.nc_x = nx_c; ws.nc_y = ny_c; ws.nc_z = nz_c
    
    nElem_c = nx_c * ny_c * nz_c
    nNodes_c = (nx_c + 1) * (ny_c + 1) * (nz_c + 1)
    
    if !ws.is_initialized || length(ws.density_coarse) != nElem_c
        if ws.is_initialized
            CUDA.unsafe_free!(ws.r_coarse); CUDA.unsafe_free!(ws.x_coarse)
            CUDA.unsafe_free!(ws.diag_coarse); CUDA.unsafe_free!(ws.density_coarse)
            CUDA.unsafe_free!(ws.conn_coarse); CUDA.unsafe_free!(ws.fine_node_coords_normalized)
            CUDA.unsafe_free!(ws.r_fine_full); CUDA.unsafe_free!(ws.z_fine_full)
        end
        ws.r_coarse = CUDA.zeros(Float32, nNodes_c * 3)
        ws.x_coarse = CUDA.zeros(Float32, nNodes_c * 3)
        ws.diag_coarse = CUDA.zeros(Float32, nNodes_c * 3)
        ws.density_coarse = CUDA.zeros(Float32, nElem_c)
        _, elems_c, _ = Mesh.generate_mesh(nx_c, ny_c, nz_c)
        ws.conn_coarse = CuArray(Int32.(vec(elems_c')))
        ws.is_initialized = true
    end
    
    nNodes_f = size(fine_nodes, 1)
    if ws.fine_node_coords_normalized === nothing || size(ws.fine_node_coords_normalized, 1) != nNodes_f
        if ws.fine_node_coords_normalized !== nothing; CUDA.unsafe_free!(ws.fine_node_coords_normalized); end
        if ws.r_fine_full !== nothing; CUDA.unsafe_free!(ws.r_fine_full); end
        if ws.z_fine_full !== nothing; CUDA.unsafe_free!(ws.z_fine_full); end
        
        ws.fine_node_coords_normalized = CUDA.zeros(Float32, nNodes_f, 3)
        ws.r_fine_full = CUDA.zeros(Float32, nNodes_f * 3)
        ws.z_fine_full = CUDA.zeros(Float32, nNodes_f * 3)
    end
    
    nodes_gpu_temp = CuArray(fine_nodes)
    @cuda threads=512 blocks=cld(nNodes_f, 512) normalize_coords_kernel!(
        ws.fine_node_coords_normalized, nodes_gpu_temp, ws.dx_c, ws.dy_c, ws.dz_c
    )
    CUDA.unsafe_free!(nodes_gpu_temp)
    
    fill!(ws.density_coarse, 0.001f0) 
    @cuda threads=512 blocks=cld(nNodes_f, 512) map_density_kernel!(
        ws.density_coarse, ws.fine_node_coords_normalized, ws.nc_x, ws.nc_y, ws.nc_z
    )
    
    fill!(ws.diag_coarse, 1.0f0)
    
    Ke_base_c = Element.get_canonical_stiffness(ws.dx_c, ws.dy_c, ws.dz_c, 0.3f0)
    Ke_diag_c = CuArray(diag(Ke_base_c))
    
    @cuda threads=512 blocks=cld(nElem_c, 512) compute_diagonal_atomic_kernel!(
        ws.diag_coarse, ws.conn_coarse, Ke_diag_c, ws.density_coarse, nElem_c
    )
    CUDA.unsafe_free!(Ke_diag_c)
    
    return ws
end

function apply_mg_vcycle!(z_fine_free, r_fine_free, ws::MGWorkspace, fine_diag_inv_free, map_gpu, n_free)
    # 1. Pre-Smoothing (Jacobi on Free DOFs)
    @. z_fine_free = r_fine_free * fine_diag_inv_free
    
    # 2. Expand Free Residual to Full Geometric Vector
    fill!(ws.r_fine_full, 0.0f0)
    @cuda threads=512 blocks=cld(n_free, 512) expand_kernel!(ws.r_fine_full, r_fine_free, map_gpu, n_free)
    
    # 3. Restriction: r_c = P^T * r_f_full
    fill!(ws.r_coarse, 0.0f0)
    nNodes_f = size(ws.fine_node_coords_normalized, 1)
    @cuda threads=512 blocks=cld(nNodes_f, 512) restrict_residual_kernel!(
        ws.r_coarse, ws.r_fine_full, ws.fine_node_coords_normalized, nNodes_f, ws.nc_x, ws.nc_y, ws.nc_z
    )
    
    # 4. Coarse Solve (Damped Jacobi)
    # x_c = D_c^-1 * r_c
    @. ws.x_coarse = ws.r_coarse / (ws.diag_coarse + 1.0f-9)
    
    # 5. Prolongation: x_f_full = P * x_c
    fill!(ws.z_fine_full, 0.0f0)
    @cuda threads=512 blocks=cld(nNodes_f, 512) prolongate_correction_kernel!(
        ws.z_fine_full, ws.x_coarse, ws.fine_node_coords_normalized, nNodes_f, ws.nc_x, ws.nc_y, ws.nc_z
    )
    
    # 6. Add Coarse Correction to Free Solution (z_fine_free += Contract(z_fine_full))
    @cuda threads=512 blocks=cld(n_free, 512) contract_add_kernel!(z_fine_free, ws.z_fine_full, map_gpu, n_free)
    
    return nothing
end

end