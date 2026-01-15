import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Dict
from functools import partial
from jax.flatten_util import ravel_pytree

# --- SOLVER LOGIC ---

def anderson_acceleration(f, z_init, max_iter, tol, m=5, beta=1.0, lam=1e-4):
    """
    Solves z = f(z) using Anderson Acceleration (Type II).
    Optimized for JAX compilation.
    """
    batch_size = z_init.shape[0]
    d = z_init.size // batch_size
    
    # State: (iteration, z_curr, f_curr, History_G, History_Z)
    # G = f(z) - z
    # We store flattened histories: (Batch, m, d)
    
    def step_fn(val):
        iter_k, z, _, G_hist, Z_hist = val
        
        # 1. Evaluate
        f_z = f(z)
        g = f_z - z
        
        z_flat = z.reshape(batch_size, -1)
        g_flat = g.reshape(batch_size, -1)
        
        # 2. Update History (Circular or Shift)
        # Shift: [0] becomes newest
        G_hist = jnp.roll(G_hist, 1, axis=1)
        Z_hist = jnp.roll(Z_hist, 1, axis=1)
        
        G_hist = G_hist.at[:, 0, :].set(g_flat)
        Z_hist = Z_hist.at[:, 0, :].set(z_flat)
        
        # 3. Compute Mixing
        # We need to solve for gamma: || g_k - Sum gamma_j (g_k - g_{k-j-1}) ||
        # Let DeltaG_j = g_k - G_hist[:, j+1] ?? 
        # Actually standard definition uses previous m residuals.
        # k=0: no history.
        # We use a mask or robust lstsq.
        
        # Construct H matrix (Batch, d, m)
        # H_j = g_flat - G_hist[:, j+1?]
        # If we just use current G_hist, index 0 is current.
        # H[:, j] = G_hist[:, 0] - G_hist[:, j+1]
        # We use m columns.
        
        # Efficient construction:
        # G_current_broad = g_flat[:, None, :] # (B, 1, d)
        # G_history_slice = G_hist[:, 1:, :] # (B, m-1, d) ? We need m diffs?
        # If history not full, we effectively zero out or use regularization.
        
        # Using a simpler form: H = G_hist[:, 1:m+1] - G_hist[:, 0] ?? 
        # Let's stick to standard Code:
        # Delta G = G_k - G_{k-j}
        
        # Shapes:
        # G_hist: (B, m, d). index 0 is k.
        # We take differences w.r.t index 0.
        # DG = G_hist[:, 0:1, :] - G_hist[:, 1:, :] # (B, m-1, d)
        
        m_eff = m # Fixed size
        
        # (B, d, m-1)
        # We transpose to (B, m-1, d) for solving G gamma = g ??
        # Target y = g_k (B, d)
        # Matrix X = DG^T ?
        # We tackle: min || g - DG * gamma ||
        
        current_g = G_hist[:, 0, :]
        prev_G = G_hist[:, 1:, :] # (B, m-1, d)
        current_z = Z_hist[:, 0, :]
        prev_Z = Z_hist[:, 1:, :]
        
        DG = current_g[:, None, :] - prev_G # (B, m-1, d)
        DZ = current_z[:, None, :] - prev_Z
        
        # Solve batch lstsq
        # DG is (B, m-1, d). We treat d as samples, m-1 as features?
        # H gamma = g.
        # H is (d, m-1).
        
        # JAX lstsq expects (..., N, M). N > M.
        # Transpose DG to (B, d, m-1)
        H = jnp.transpose(DG, (0, 2, 1)) 
        y = current_g[:, :, None] # (B, d, 1)
        
        # Ridge regression: (H^T H + lam I) gamma = H^T y
        # or straight lstsq with rcond
        
        # Optimization: m is small (5), d is large (256*64).
        # Compute Gram matrix H^T H -> (B, m-1, m-1)
        HTH = jnp.matmul(jnp.transpose(H, (0, 2, 1)), H) 
        HTH = HTH + lam * jnp.eye(m-1)[None, :, :]
        
        HTy = jnp.matmul(jnp.transpose(H, (0, 2, 1)), y) # (B, m-1, 1)
        
        # Solve small system
        # Metal-Optimized 4x4 Block Inversion
        # Implements strict Schur complement inversion to bypass missing linear algebra kernels.
        def inv2x2(M):
            # M: (B, 2, 2)
            a, b = M[..., 0, 0], M[..., 0, 1]
            c, d = M[..., 1, 0], M[..., 1, 1]
            det = a * d - b * c
            invDet = 1.0 / (det + 1e-6)
            # Row 0: d, -b
            # Row 1: -c, a
            # Stack manually
            row0 = jnp.stack([d, -b], axis=-1)
            row1 = jnp.stack([-c, a], axis=-1)
            return jnp.stack([row0, row1], axis=-2) * invDet[..., None, None]

        def inv4x4(M):
            # Blockwise inversion: [[A, B], [C, D]]
            A = M[..., 0:2, 0:2]
            B = M[..., 0:2, 2:4]
            C = M[..., 2:4, 0:2]
            D = M[..., 2:4, 2:4]
            
            A_inv = inv2x2(A)
            # Schur complement of A: S = D - C A_ipv B
            CAinv = jnp.matmul(C, A_inv)
            S = D - jnp.matmul(CAinv, B)
            S_inv = inv2x2(S)
            
            # Result blocks
            # inv(M) = [[Ainv + Ainv B Sinv C Ainv, -Ainv B Sinv], [-Sinv C Ainv, Sinv]]
            SinvCAinv = jnp.matmul(S_inv, CAinv)
            
            block11 = A_inv + jnp.matmul(jnp.matmul(A_inv, B), SinvCAinv)
            block12 = -jnp.matmul(jnp.matmul(A_inv, B), S_inv)
            block21 = -SinvCAinv
            block22 = S_inv
            
            # Concatenate
            top = jnp.concatenate([block11, block12], axis=-1)
            bot = jnp.concatenate([block21, block22], axis=-1)
            return jnp.concatenate([top, bot], axis=-2)

        # HTH is (B, m-1, m-1). With m=5, this is exactly 4x4.
        gamma = jnp.matmul(inv4x4(HTH), HTy) # (B, m-1, 1)
        gamma = gamma[:, :, 0] # (B, m-1)
        
        # 4. Update
        # z_new = z_k - Sum gamma_j (z_k - z_{k-j-1} + g_k - g_{k-j-1}) ??
        # Standard: z_new = z_k + beta * g_k - Sum gamma_j ( (z_k - z_{prev}) + beta * (g_k - g_{prev}) )
        
        correction = jnp.sum(gamma[:, :, None] * (DZ + beta * DG), axis=1) # (B, d)
        
        z_new = (z_flat + beta * current_g) - correction
        
        # Fallback for early iterations (k < m) or instability (gamma huge)
        # Heuristic: if k < m, gamma zeros? Or just let ridge handle it?
        # With zero initialization of history, DG will be g_k - 0 = g_k.
        # It handles itself via regularization usually.
        # Guard:
        is_early = iter_k < m
        # If early, use simple Damped
        z_damped = z_flat + beta * current_g
        
        z_out_flat = jnp.where(is_early, z_damped, z_new)
        z_out = z_out_flat.reshape(z.shape)
        
        # Calculate diff BEFORE update? Or between steps?
        # DEQ usually checks || f(z) - z || i.e. norm(g)
        diff = jnp.mean(jnp.abs(g))
        
        return iter_k + 1, z_out, f_z, G_hist, Z_hist

    def cond_fn(val):
        iter_k, _, _, _, _ = val
        # To inspect convergence, we need the diff.
        # We can pass it in state or recompute.
        # Recomputing is safest for 'cond'.
        # But we optimized step_fn to compute f(z).
        # Let's assume step_fn does valid update.
        return iter_k < max_iter

    # Init
    batch_size = z_init.shape[0]
    flat_dim = z_init.size // batch_size
    
    Z_hist = jnp.zeros((batch_size, m, flat_dim))
    G_hist = jnp.zeros((batch_size, m, flat_dim))
    
    # We need f_curr for state
    f_init = f(z_init)
    
    # (iter, z, f_z, G_hist, Z_hist)
    init_val = (0, z_init, f_init, G_hist, Z_hist)
    
    # Unrolled execution for deterministic graph compilation.
    # Enables full compiler optimization and avoids dynamic control flow overhead.
    val = init_val
    for i in range(max_iter):
        val = step_fn(val)
        
    res = val
    return res[1]

# Alias
fixed_point_iteration = anderson_acceleration


# --- CUSTOM VJP DEFINITION ---

@partial(jax.custom_vjp, nondiff_argnums=(0, 4))
def deq_fixed_point(f_apply, params, x_input, z_init, solver_kwargs):
    def f_fixed(z):
        return f_apply(params, z, x_input)
    z_star = fixed_point_iteration(f_fixed, z_init, **solver_kwargs)
    return z_star

def deq_fixed_point_fwd(f_apply, params, x_input, z_init, solver_kwargs):
    z_star = deq_fixed_point(f_apply, params, x_input, z_init, solver_kwargs)
    return z_star, (params, x_input, z_star)

def deq_fixed_point_bwd(f_apply, solver_kwargs, res, g):
    params, x_input, z_star = res
    grad_z_star = g
    
    # Fixed point equation for adjoint u: u = grad_z * J + g (or u = uJ + g)
    # Using VJP: vjp(u) = u * J
    # We solve u = vjp(u) + g
    
    def bwd_f(u):
        _, vjp_fun = jax.vjp(lambda z: f_apply(params, z, x_input), z_star)
        # vjp_fun(u) computes u^T J
        ju = vjp_fun(u)[0]
        return grad_z_star + ju
        
    # Adjoint solve usually doesn't need dampening, but stability helps.
    # We use same solver args
    u_star = fixed_point_iteration(bwd_f, jnp.zeros_like(grad_z_star), **solver_kwargs)
    
    # Gradients w.r.t params and x
    _, vjp_fun_params_x = jax.vjp(lambda p, x: f_apply(p, z_star, x), params, x_input)
    grads_params, grads_x = vjp_fun_params_x(u_star)
    
    return grads_params, grads_x, None

deq_fixed_point.defvjp(deq_fixed_point_fwd, deq_fixed_point_bwd)

# --- MODULE ---

# --- MODULE ---

class DeepEquilibriumModel(nn.Module):
    """
    Deep Equilibrium Model (DEQ) with O(1) Memory Training.
    
    This module implements the Implicit Differentiation utilizing `jax.custom_vjp`.
    
    Key Mechanics:
    1.  **Block Instantiation**: The underlying layer (e.g., UniversalTransformerBlock) is defined
        conceptually. Its parameters are managed explicitly to allow passing them into the
        stateless fixed-point solver.
    2.  **Fixed Point Solver**: Finds z* such that z* = f(z*, x) using Anderson Acceleration
        or simple iteration.
    3.  **Implicit Backpropagation**: Instead of unrolling the graph (O(T) memory), we solve
        for the adjoint vector using the Inverse Jacobian at the fixed point (O(1) memory).
    
    Attributes:
        block_class: The Flax Module class for the repeating layer.
        block_args: Arguments to initialize the block.
        max_iter: Maximum iterations for the forward solver.
        tol: Tolerance for convergence.
    """
    block_class: Any
    block_args: Dict[str, Any]
    max_iter: int = 12
    tol: float = 1e-4
    beta: float = 1.0 # Anderson mixing beta / Dampening factor

    @nn.compact
    def __call__(self, x_input):
        # We need to explicitly manage the submodule parameters to pass them to custom_vjp.
        # Standard Flax `self.param` works for weights, but here we need to manage 
        # a whole submodule's tree of parameters as a single unit or collection 
        # to pass to the functional 'f_apply'.

        # 1. Define the initialization function for the block
        block = self.block_class(**self.block_args)
        
        def init_block_params(rng):
            # Run a dummy init pass to generate the parameter structure
            # z_init is zeros, x_input is real structure
            z_dummy = jnp.zeros_like(x_input)
            variables = block.init(rng, z_dummy, x_input, train=True)
            return variables['params'] # Extract just the params dictionary

        # 2. Declare the block parameters as a Flax variable collection.
        # Use lazy RNG fetching to avoid InvalidRngError during inference (when params exist but RNG doesn't)
        rng = None
        if not self.has_variable('params', 'block_params'):
            rng = self.make_rng('params')
            
        block_vars = self.variable('params', 'block_params', init_block_params, rng)
        
        # 3. Define the functional application wrapper
        # This function 'f' must be pure: f(params, z, x) -> z_next
        def apply_fn(p, z, x):
            # We use block.apply with the specific params passed by the solver/VJP
            return block.apply({'params': p}, z, x, train=True)

        # 4. Prepare Logic
        z_init = jnp.zeros_like(x_input)
        solver_args = {'max_iter': self.max_iter, 'tol': self.tol, 'beta': self.beta, 'm': 5}
        
        # 5. Execute DEQ with Custom VJP
        # We pass block_vars.value (the dict) as the 'params' argument.
        # JAX handles PyTrees (dicts) transparently in differentiation.
        z_star = deq_fixed_point(apply_fn, block_vars.value, x_input, z_init, solver_args)
        
        return z_star

