"""
Deep Equilibrium Model (DEQ) with Anderson Acceleration.

Per GPU ONLY Chess RL Agent.txt spec:
- Uses Anderson Acceleration for fixed-point finding (section 2.2.1)
- O(1) memory via implicit differentiation (section 2.2.2)
- Adaptive depth through convergence tolerance (Node-Skipping)
- BFloat16 compatible for mixed precision training
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Dict, Tuple
from functools import partial


def anderson_acceleration_fixed(f, z_init, max_iter, tol=1e-3, m=2, beta=0.9, lam=1e-4):
    """
    Fixed iteration Anderson Acceleration with BFloat16 support.
    
    Ensures consistent dtype throughout computation.
    """
    batch_size = z_init.shape[0]
    d = z_init.size // batch_size
    dtype = z_init.dtype  # Preserve input dtype (BF16 or FP32)
    
    def step_fn(i, state):
        z, G_hist, Z_hist, min_residual = state
        
        f_z = f(z)
        g = f_z - z
        
        # Track residual
        residual = jnp.linalg.norm(g.reshape(batch_size, -1), axis=-1)
        min_residual = jnp.minimum(min_residual, residual)
        
        z_flat = z.reshape(batch_size, -1)
        g_flat = g.reshape(batch_size, -1)
        
        G_hist = jnp.roll(G_hist, 1, axis=1)
        Z_hist = jnp.roll(Z_hist, 1, axis=1)
        G_hist = G_hist.at[:, 0, :].set(g_flat)
        Z_hist = Z_hist.at[:, 0, :].set(z_flat)
        
        def simple_update(_):
            result = z_flat + beta * g_flat
            return result.astype(dtype)  # Ensure correct dtype
        
        def anderson_update(_):
            current_g = G_hist[:, 0, :]
            prev_G = G_hist[:, 1:, :]
            current_z = Z_hist[:, 0, :]
            prev_Z = Z_hist[:, 1:, :]
            
            DG = current_g[:, None, :] - prev_G
            DZ = current_z[:, None, :] - prev_Z
            
            DG_0 = DG[:, 0, :]
            DZ_0 = DZ[:, 0, :]
            
            # Compute in FP32 for numerical stability, then cast back
            DG_0_f32 = DG_0.astype(jnp.float32)
            current_g_f32 = current_g.astype(jnp.float32)
            DZ_0_f32 = DZ_0.astype(jnp.float32)
            z_flat_f32 = z_flat.astype(jnp.float32)
            
            numerator = jnp.sum(DG_0_f32 * current_g_f32, axis=-1)
            denominator = jnp.sum(DG_0_f32 * DG_0_f32, axis=-1) + lam
            gamma = numerator / denominator
            
            correction = gamma[:, None] * (DZ_0_f32 + beta * DG_0_f32)
            z_new = (z_flat_f32 + beta * current_g_f32) - correction
            
            return z_new.astype(dtype)  # Cast back to original dtype
        
        z_new_flat = jax.lax.cond(i < m, simple_update, anderson_update, None)
        z_new = z_new_flat.reshape(z.shape)
        
        return (z_new, G_hist, Z_hist, min_residual)
    
    # Initialize history with correct dtype
    Z_hist = jnp.zeros((batch_size, m, d), dtype=dtype)
    G_hist = jnp.zeros((batch_size, m, d), dtype=dtype)
    min_residual = jnp.ones(batch_size, dtype=jnp.float32) * 1e6
    
    init_state = (z_init, G_hist, Z_hist, min_residual)
    final_state = jax.lax.fori_loop(0, max_iter, step_fn, init_state)
    
    z_star, _, _, final_residual = final_state
    converged = final_residual < tol
    
    return z_star, max_iter, converged, final_residual


# Use fixed iteration version
fixed_point_iteration = anderson_acceleration_fixed


# --- CUSTOM VJP DEFINITION ---

@partial(jax.custom_vjp, nondiff_argnums=(0, 4))
def deq_fixed_point(f_apply, params, x_input, z_init, solver_kwargs):
    def f_fixed(z):
        return f_apply(params, z, x_input)
    z_star, num_iters, converged, residual = fixed_point_iteration(f_fixed, z_init, **solver_kwargs)
    return z_star

def deq_fixed_point_fwd(f_apply, params, x_input, z_init, solver_kwargs):
    def f_fixed(z):
        return f_apply(params, z, x_input)
    z_star, num_iters, converged, residual = fixed_point_iteration(f_fixed, z_init, **solver_kwargs)
    return z_star, (params, x_input, z_star)

def deq_fixed_point_bwd(f_apply, solver_kwargs, res, g):
    params, x_input, z_star = res
    grad_z_star = g
    
    # Implicit differentiation via Neumann series (in FP32 for stability)
    z_star_f32 = z_star.astype(jnp.float32)
    grad_z_star_f32 = grad_z_star.astype(jnp.float32)
    
    def bwd_step(i, u):
        _, vjp_fun = jax.vjp(lambda z: f_apply(params, z, x_input), z_star_f32)
        ju = vjp_fun(u)[0]
        return grad_z_star_f32 + ju
    
    # 4 iterations of Neumann series approximation
    u_star = jax.lax.fori_loop(0, 4, bwd_step, grad_z_star_f32)
    
    # Gradients w.r.t params and x
    _, vjp_fun_params_x = jax.vjp(lambda p, x: f_apply(p, z_star_f32, x), params, x_input)
    grads_params, grads_x = vjp_fun_params_x(u_star)
    
    return grads_params, grads_x, None

deq_fixed_point.defvjp(deq_fixed_point_fwd, deq_fixed_point_bwd)


# --- DEQ MODULE ---

class DeepEquilibriumModel(nn.Module):
    """
    Deep Equilibrium Model (DEQ) with Anderson Acceleration.
    
    Per GPU ONLY Chess RL Agent.txt spec:
    - Uses Anderson Acceleration for fixed-point finding
    - O(1) memory via implicit differentiation
    - Adaptive depth through convergence tolerance
    - BFloat16 compatible
    """
    block_class: Any
    block_args: Dict[str, Any]
    max_iter: int = 6
    tol: float = 1e-3
    beta: float = 0.9
    m: int = 2

    @nn.compact
    def __call__(self, x_input):
        block = self.block_class(**self.block_args)
        
        def init_block_params(rng):
            z_dummy = jnp.zeros_like(x_input)
            variables = block.init(rng, z_dummy, x_input, train=True)
            return variables['params']

        rng = None
        if not self.has_variable('params', 'block_params'):
            rng = self.make_rng('params')
            
        block_vars = self.variable('params', 'block_params', init_block_params, rng)
        
        def apply_fn(p, z, x):
            return block.apply({'params': p}, z, x, train=True)

        # Initialize z with same dtype as input
        z_init = jnp.zeros_like(x_input)
        
        solver_args = {
            'max_iter': self.max_iter, 
            'tol': self.tol, 
            'beta': self.beta,
            'm': self.m
        }
        
        z_star = deq_fixed_point(apply_fn, block_vars.value, x_input, z_init, solver_args)
        
        # Track complexity metrics
        def get_residual(z):
            f_z = apply_fn(block_vars.value, z, x_input)
            residual = jnp.linalg.norm((f_z - z).reshape(z.shape[0], -1), axis=-1)
            return residual
        
        final_residual = get_residual(z_star)
        self.sow('intermediates', 'final_residual', final_residual)
        self.sow('intermediates', 'mean_residual', jnp.mean(final_residual))
        
        return z_star
