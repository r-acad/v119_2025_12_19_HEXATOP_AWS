# FILE: .\src\Solvers\GPUSolver.jl

module GPUSolver

using LinearAlgebra, Printf
using CUDA
using SparseArrays
using Base.Threads
using Dates 
using Statistics
using ..Element
using ..GPUGeometricMultigrid 
using ..Diagnostics
using ..Helpers

export solve_system_gpu

const C_RESET   = "\u001b[0m"
const C_BOLD    = "\u001b[1m"
const C_CYAN    = "\u001b[36m"
const C_GREEN   = "\u001b[32m"
const C_YELLOW  = "\u001b[33m"
const C_RED     = "\u001b[31m"

const HISTORY_LOG_FILE = "convergence_history_log.txt"

function dual_log(msg::String; force_file::Bool=false)
    print(msg)
    if force_file
        clean_msg = replace(msg, r"\u001b\[[0-9;]*m" => "")
        try
            open(HISTORY_LOG_FILE, "a") do io; write(io, clean_msg); end
        catch e; @warn "Log write failed: $e"; end
    end
end

function log_section_header(title::String, outer_iter::Any="?")
    width = 80
    s = "\n" * C_CYAN * "="^width * C_RESET * "\n"
    full_title = "$title [Topo Opt Iter: $outer_iter]"
    pad = max(0, (width - length(full_title) - 2) ÷ 2)
    s *= " "^pad * C_BOLD * full_title * C_RESET * "\n"
    s *= C_CYAN * "="^width * C_RESET * "\n"
    dual_log(s; force_file=true)
end

mutable struct CGWorkspace
    is_initialized::Bool
    precision_type::DataType
    r::Any 
    p::Any
    z_Ap::Any 
    x::Any    
    
    conn_gpu::Any        
    factors_gpu::Any      
    Ke_gpu::Any           
    map_gpu::Any          

    CGWorkspace() = new(false, Float32, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

const GLOBAL_CG_CACHE = CGWorkspace()

function get_cg_workspace(n_free::Int, n_total_nodes::Int, n_elem::Int, T::DataType)
    ws = GLOBAL_CG_CACHE
    if !ws.is_initialized || length(ws.r) != n_free || length(ws.factors_gpu) != n_elem
        if ws.is_initialized
            CUDA.unsafe_free!(ws.r); CUDA.unsafe_free!(ws.p); CUDA.unsafe_free!(ws.z_Ap); CUDA.unsafe_free!(ws.x)
            CUDA.unsafe_free!(ws.conn_gpu); CUDA.unsafe_free!(ws.factors_gpu); CUDA.unsafe_free!(ws.Ke_gpu); CUDA.unsafe_free!(ws.map_gpu)
            dual_log("  [GPU Memory] Resizing Workspace (High-Capacity Int64 Mode)...\n"; force_file=true)
        end
        ws.r = CUDA.zeros(T, n_free)
        ws.p = CUDA.zeros(T, n_free)
        ws.z_Ap = CUDA.zeros(T, n_free) 
        ws.x = CUDA.zeros(T, n_free)
        
        # Connectivity is flattened Int
        ws.conn_gpu = CUDA.zeros(Int, n_elem * 8)
        
        ws.factors_gpu = CUDA.zeros(T, n_elem)
        ws.Ke_gpu = CUDA.zeros(T, 24, 24)
        ws.map_gpu = CUDA.zeros(Int, n_free) 
        ws.precision_type = T
        ws.is_initialized = true
    end
    fill!(ws.r, T(0.0)); fill!(ws.p, T(0.0)); fill!(ws.z_Ap, T(0.0)); fill!(ws.x, T(0.0))
    return ws
end

function matvec_unstructured_atomic_kernel!(y_full, x_full, conn, Ke, factors, nElem)
    # Each thread handles one element
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if e <= nElem
        base_idx = (e - 1) * 8
        n = (conn[base_idx+1], conn[base_idx+2], conn[base_idx+3], conn[base_idx+4],
             conn[base_idx+5], conn[base_idx+6], conn[base_idx+7], conn[base_idx+8])

        u_loc = (
            x_full[(n[1]-1)*3+1], x_full[(n[1]-1)*3+2], x_full[(n[1]-1)*3+3],
            x_full[(n[2]-1)*3+1], x_full[(n[2]-1)*3+2], x_full[(n[2]-1)*3+3],
            x_full[(n[3]-1)*3+1], x_full[(n[3]-1)*3+2], x_full[(n[3]-1)*3+3],
            x_full[(n[4]-1)*3+1], x_full[(n[4]-1)*3+2], x_full[(n[4]-1)*3+3],
            x_full[(n[5]-1)*3+1], x_full[(n[5]-1)*3+2], x_full[(n[5]-1)*3+3],
            x_full[(n[6]-1)*3+1], x_full[(n[6]-1)*3+2], x_full[(n[6]-1)*3+3],
            x_full[(n[7]-1)*3+1], x_full[(n[7]-1)*3+2], x_full[(n[7]-1)*3+3],
            x_full[(n[8]-1)*3+1], x_full[(n[8]-1)*3+2], x_full[(n[8]-1)*3+3]
        )

        factor = factors[e]

        @inbounds for i in 1:24
            val = zero(u_loc[1])
            val += Ke[i, 1]*u_loc[1] + Ke[i, 2]*u_loc[2] + Ke[i, 3]*u_loc[3] + Ke[i, 4]*u_loc[4]
            val += Ke[i, 5]*u_loc[5] + Ke[i, 6]*u_loc[6] + Ke[i, 7]*u_loc[7] + Ke[i, 8]*u_loc[8]
            val += Ke[i, 9]*u_loc[9] + Ke[i, 10]*u_loc[10] + Ke[i, 11]*u_loc[11] + Ke[i, 12]*u_loc[12]
            val += Ke[i, 13]*u_loc[13] + Ke[i, 14]*u_loc[14] + Ke[i, 15]*u_loc[15] + Ke[i, 16]*u_loc[16]
            val += Ke[i, 17]*u_loc[17] + Ke[i, 18]*u_loc[18] + Ke[i, 19]*u_loc[19] + Ke[i, 20]*u_loc[20]
            val += Ke[i, 21]*u_loc[21] + Ke[i, 22]*u_loc[22] + Ke[i, 23]*u_loc[23] + Ke[i, 24]*u_loc[24]
            
            target_node = n[(i - 1) ÷ 3 + 1]
            target_idx = (target_node - 1) * 3 + (i - 1) % 3 + 1
            CUDA.atomic_add!(pointer(y_full, target_idx), val * factor)
        end
    end
    return nothing
end

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

function expand_kernel!(x_full, x_free, map, n_free)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_full[map[idx]] = x_free[idx]
    end
    return nothing
end

function contract_kernel!(y_free, y_full, map, n_free)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds y_free[idx] = y_full[map[idx]]
    end
    return nothing
end

function jacobi_precond_kernel!(z, r, M_inv, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds z[idx] = r[idx] * M_inv[idx]
    end
    return nothing
end

function get_free_dofs(bc_indicator)
    nNodes = size(bc_indicator, 1)
    ndof = nNodes * 3
    constrained = falses(ndof)
    for i in 1:nNodes
        if bc_indicator[i,1] > 0; constrained[3*(i-1)+1] = true; end
        if bc_indicator[i,2] > 0; constrained[3*(i-1)+2] = true; end
        if bc_indicator[i,3] > 0; constrained[3*(i-1)+3] = true; end
    end
    return findall(!, constrained)
end

function gpu_matrix_free_cg_solve(nodes, elements, E, nu, bc, f, density;
                                  max_iter=40000, tol=1e-6, shift_factor=1.0e-4,
                                  min_stiffness_threshold=1.0e-3, u_guess=[], config=Dict())
    
    CUDA.allowscalar(false)
    gpu_profile = get(config, "hardware_profile_applied", "RTX")
    use_double = (gpu_profile == "H200" || gpu_profile == "V100" || gpu_profile == "A100") || get(config, "force_float64", false)
    T = use_double ? Float64 : Float32
    
    outer_iter = get(config, "current_outer_iter", "?")
    
    # === START OF NEW JOB LOGGING ===
    if outer_iter == 1 || outer_iter == "1"
        try
            open(HISTORY_LOG_FILE, "a") do io 
                write(io, "\n\n" * "-"^80 * "\n")
                write(io, "NEW JOB SESSION STARTED: $(Dates.now())\n")
                write(io, "-"^80 * "\n")
                write(io, "================================================================================\n")
                write(io, "HEXA FEM SOLVER CONVERGENCE HISTORY\n")
                write(io, "================================================================================\n\n")
                write(io, Diagnostics.get_hardware_info())
                write(io, "\n")
                write(io, "--- CONFIGURATION SNAPSHOT ---\n")
                Diagnostics.log_full_config(io, config)
                write(io, "\n" * "="^80 * "\n")
            end
        catch e
            @warn "Failed to write history log header: $e"
        end
    end
    # === END OF NEW JOB LOGGING ===

    solver_params = get(config, "solver_parameters", Dict())
    stagnation_tol = T(get(solver_params, "stagnation_tolerance", 0.0))
    max_shift_attempts = Int(get(solver_params, "max_shift_attempts", 3))
    shift_multiplier = T(get(solver_params, "shift_multiplier", 10.0))
    
    nNodes = size(nodes, 1)
    nElem = size(elements, 1)
    nDof = nNodes * 3
    free_dofs = get_free_dofs(bc)
    n_free = length(free_dofs)
    
    geom_conf = config["geometry"]
    dx = Float32(get(geom_conf, "dx_computed", 1.0))
    dy = Float32(get(geom_conf, "dy_computed", 1.0))
    dz = Float32(get(geom_conf, "dz_computed", 1.0))
    
    want_mg = (get(solver_params, "preconditioner", "jacobi") == "multigrid")
    
    # === PRE-FLIGHT MEMORY CHECK ===
    Helpers.enforce_gpu_memory_safety(nElem, nNodes, use_double, want_mg)
    # ===============================

    if CUDA.functional()
        free_mem, tot_mem = CUDA.available_memory(), CUDA.total_memory()
        
        log_section_header("GPU SOLVER (ATOMIC)", outer_iter)
        dual_log(@sprintf("  Nodes: %d | Elems: %d | Free DOFs: %d\n", nNodes, nElem, n_free); force_file=true)
        dual_log(@sprintf("  Element Size: dx=%.4f, dy=%.4f, dz=%.4f\n", dx, dy, dz); force_file=true)
        dual_log(@sprintf("  Pre-Alloc VRAM: %.2f GB Free / %.2f GB Total\n", free_mem/1024^3, tot_mem/1024^3); force_file=true)
    end

    ws = get_cg_workspace(n_free, nNodes, nElem, T)
    
    Ke_base = Element.get_canonical_stiffness(dx, dy, dz, Float32(nu))
    copyto!(ws.Ke_gpu, Matrix{T}(Ke_base))
    
    conn_flat = vec(elements') 
    copyto!(ws.conn_gpu, CuArray(conn_flat))
    
    fact_vec = Vector{T}(E .* density)
    copyto!(ws.factors_gpu, CuArray(fact_vec))
    copyto!(ws.map_gpu, CuArray(free_dofs)) 
    
    mg_ws = nothing
    if want_mg
        try
            mg_ws = GPUGeometricMultigrid.setup_multigrid(nodes, density, config)
            dual_log(C_GREEN * "  [MG Init] Multigrid Levels Initialized Successfully.\n" * C_RESET; force_file=true)
        catch e
            dual_log(C_YELLOW * "  [MG Init Failed] Error: $e. Falling back to Jacobi.\n" * C_RESET; force_file=true)
            want_mg = false
            GC.gc(); CUDA.reclaim()
        end
    end
    
    # === CRITICAL POST-ALLOCATION CHECK ===
    if CUDA.functional()
        free_mem_post, _ = CUDA.available_memory(), CUDA.total_memory()
        dual_log(@sprintf("  Post-Alloc VRAM: %.2f GB Free\n", free_mem_post/1024^3); force_file=true)
        
        # STOP if less than 100MB is free.
        # This prevents the "Freeze" by catching it before the compute kernel launches.
        if free_mem_post < 100 * 1024 * 1024
            dual_log(C_RED * "  >>> [CRITICAL] VRAM EXHAUSTED (<100MB). Aborting to prevent freeze.\n" * C_RESET; force_file=true)
            error("GPU VRAM Exhausted. Reduce element count or switch to Jacobi preconditioner.")
        end
    end
    # ======================================

    x_gpu, r_gpu, p_gpu = ws.x, ws.r, ws.p
    Ap_free = ws.z_Ap 
    z_gpu = ws.z_Ap      
    conn_gpu = ws.conn_gpu
    factors_gpu = ws.factors_gpu
    Ke_gpu = ws.Ke_gpu
    map_gpu = ws.map_gpu
    
    b_gpu = CuVector{T}(f[free_dofs])
    x_full = CUDA.zeros(T, nDof)
    Ap_full = CUDA.zeros(T, nDof)
    diag_full = CUDA.zeros(T, nDof)
    
    Ke_diag_cpu = diag(Ke_base)
    Ke_diag_gpu = CuArray{T}(Ke_diag_cpu)
    
    threads_per_block = 256
    @cuda threads=threads_per_block blocks=cld(nElem, threads_per_block) compute_diagonal_atomic_kernel!(
        diag_full, conn_gpu, Ke_diag_gpu, factors_gpu, nElem
    )
    
    diag_free = CUDA.zeros(T, n_free)
    @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) contract_kernel!(diag_free, diag_full, map_gpu, n_free)
    
    max_diag = maximum(diag_free)
    
    if isnan(max_diag) || isinf(max_diag)
        dual_log(C_RED * "  >>> [CRITICAL ERROR] Stiffness Matrix Diagonal contains NaN/Inf.\n" * C_RESET; force_file=true)
        CUDA.unsafe_free!(x_full); CUDA.unsafe_free!(Ap_full); CUDA.unsafe_free!(diag_full)
        CUDA.unsafe_free!(Ke_diag_gpu)
        return (zeros(T, length(f)), NaN, "Failed_Corrupt_Input")
    end
    
    M_inv = CUDA.zeros(T, n_free)
    norm_b = norm(b_gpu)
    best_x = zeros(T, n_free)
    final_rel_res = 0.0
    
    function apply_A!(y_f, x_f)
        fill!(x_full, T(0.0)) 
        @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) expand_kernel!(x_full, x_f, map_gpu, n_free)
        fill!(Ap_full, T(0.0)) 
        @cuda threads=threads_per_block blocks=cld(nElem, threads_per_block) matvec_unstructured_atomic_kernel!(
            Ap_full, x_full, conn_gpu, Ke_gpu, factors_gpu, nElem
        )
        @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) contract_kernel!(y_f, Ap_full, map_gpu, n_free)
    end

    current_use_mg = want_mg
    global_solve_success = false
    
    while true
        precond_name = current_use_mg ? "Geometric Multigrid (V-Cycle)" : "Jacobi"
        dual_log(@sprintf("  Preconditioner: %s\n", precond_name); force_file=true)
        cur_shift = shift_factor
        solve_ok = false
        fallback_to_jacobi_immediate = false
        
        for attempt in 1:max_shift_attempts
            real_shift = cur_shift * max_diag
            dual_log(@sprintf("\n  >>> ATTEMPT %d | Shift: %.1e\n", attempt, real_shift); force_file=true)
            diverged = false
            
            if !isempty(u_guess) && attempt == 1 && current_use_mg == want_mg
                copyto!(x_gpu, CuVector{T}(u_guess[free_dofs]))
            elseif attempt > 1 || current_use_mg != want_mg
                fill!(x_gpu, T(0.0))
            end
            
            M_inv .= T(1.0) ./ (diag_free .+ real_shift)
            apply_A!(Ap_free, x_gpu)
            Ap_free .+= real_shift .* x_gpu
            r_gpu .= b_gpu .- Ap_free
            
            if current_use_mg
                GPUGeometricMultigrid.apply_mg_vcycle!(z_gpu, r_gpu, mg_ws, M_inv, map_gpu, n_free)
            else
                @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) jacobi_precond_kernel!(z_gpu, r_gpu, M_inv, n_free)
            end
            
            p_gpu .= z_gpu
            rz = dot(r_gpu, z_gpu)
            t_cg = time()
            best_rel = Inf
            dual_log(@sprintf("  %s%8s %12s %12s %10s%s\n", C_BOLD, "Iter", "Res", "RelRes", "Time", C_RESET); force_file=true)
            
            for k in 1:max_iter
                apply_A!(Ap_free, p_gpu)
                Ap_free .+= real_shift .* p_gpu
                denom = dot(p_gpu, Ap_free)
                if abs(denom) < 1e-20; break; end
                alpha = rz / denom
                x_gpu .+= alpha .* p_gpu
                r_gpu .-= alpha .* Ap_free
                
                if k % 10 == 0 || k == 1
                    res = sqrt(dot(r_gpu, r_gpu))
                    rel = res / norm_b
                    final_rel_res = rel
                    
                    if rel < best_rel; best_rel = rel; copyto!(best_x, x_gpu); end
                    if rel < tol; solve_ok = true; break; end
                    
                    if isnan(rel) || isinf(rel)
                        dual_log(C_RED * "  >>> NaN/Inf DETECTED at iter $k. Aborting attempt.\n" * C_RESET; force_file=true)
                        if current_use_mg
                            fallback_to_jacobi_immediate = true
                        else
                            diverged = true
                        end
                        break
                    end

                    if rel > 10.0 || (k > 500 && rel > best_rel * 100.0)
                        dual_log(C_RED * "  >>> DIVERGENCE DETECTED at iter $k.\n" * C_RESET; force_file=true)
                        if current_use_mg; fallback_to_jacobi_immediate = true; else; diverged = true; end
                        break
                    end
                    
                    if k % 500 == 0 || k == 1
                        color = rel < tol ? C_GREEN : (rel < 0.1 ? C_YELLOW : C_RESET)
                        dual_log(@sprintf("  %s%8d %12.4e %12.4e %10.3f%s\n", color, k, res, rel, time() - t_cg, C_RESET); force_file=true)
                    end
                end
                
                if current_use_mg
                    GPUGeometricMultigrid.apply_mg_vcycle!(z_gpu, r_gpu, mg_ws, M_inv, map_gpu, n_free)
                else
                    @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) jacobi_precond_kernel!(z_gpu, r_gpu, M_inv, n_free)
                end
                rz_new = dot(r_gpu, z_gpu)
                beta = rz_new / rz
                p_gpu .= z_gpu .+ beta .* p_gpu
                rz = rz_new
            end
            
            if fallback_to_jacobi_immediate; break; end
            if solve_ok; global_solve_success = true; break; end
            if diverged
                if attempt == max_shift_attempts
                else; cur_shift *= shift_multiplier; continue; end
            end
            if stagnation_tol > 0.0 && best_rel < stagnation_tol
                solve_ok = true; global_solve_success = true; copyto!(x_gpu, best_x)
                dual_log(@sprintf("  [Adaptive] Stagnated below tolerance. Accepting.\n"); force_file=true)
                break
            end
            cur_shift *= shift_multiplier
        end
        
        if fallback_to_jacobi_immediate
            dual_log(C_YELLOW * "  >>> RESTARTING SOLVE WITH JACOBI PRECONDITIONER <<<\n" * C_RESET; force_file=true)
            current_use_mg = false; continue
        end

        if global_solve_success; break; else
            dual_log(C_RED * "  [Failure] Solver failed to reach strict tolerance.\n" * C_RESET; force_file=true)
            copyto!(x_gpu, best_x)
            break
        end
    end
    
    fill!(x_full, T(0.0))
    @cuda threads=threads_per_block blocks=cld(n_free, threads_per_block) expand_kernel!(x_full, x_gpu, map_gpu, n_free)
    x_final_full = Array(x_full) 
    
    x_stats = x_final_full[free_dofs]
    min_x = minimum(x_stats)
    max_x = maximum(x_stats)
    mean_x = mean(x_stats)
    l2_x = norm(x_stats)
    
    dual_log(@sprintf("  [Stats] Solution Range: [%.3e, %.3e]\n", min_x, max_x); force_file=true)
    dual_log(@sprintf("  [Stats] Mean: %.3e | L2 Norm: %.3e\n", mean_x, l2_x); force_file=true)
    
    CUDA.unsafe_free!(x_full); CUDA.unsafe_free!(Ap_full); CUDA.unsafe_free!(diag_full)
    CUDA.unsafe_free!(M_inv); CUDA.unsafe_free!(Ke_diag_gpu)
    
    dual_log(C_CYAN * "-"^80 * C_RESET * "\n"; force_file=true)
    final_method_name = current_use_mg ? "GMG_GPU" : "Jacobi_GPU"
    return (x_final_full, final_rel_res, final_method_name)
end

function solve_system_gpu(nodes, elements, E, nu, bc, f, density;
                          max_iter=40000, tol=1e-6, method=:native, solver=:cg, use_precond=true,
                          shift_factor=1.0e-4, min_stiffness_threshold=1.0e-3, u_guess=[], config=Dict())
    if !CUDA.functional()
        error("CUDA not functional!")
    end
    return gpu_matrix_free_cg_solve(nodes, elements, E, nu, bc, f, density;
                                    max_iter=max_iter, tol=tol, shift_factor=shift_factor,
                                    min_stiffness_threshold=min_stiffness_threshold,
                                    u_guess=u_guess, config=config)
end

end