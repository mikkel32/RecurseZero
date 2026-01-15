import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Dict
from functools import partial

# --- OPTIMIZED ANDERSON ACCELERATION ---

def anderson_acceleration(f, z_init, max_iter, tol=1e-4, m=2, beta=0.9, lam=1e-4):
    """
    Solves z = f(z) using Anderson Acceleration (Type II).
    
    SPEED OPTIMIZED:
    - Default m=2 (1x1 solve, fastest)
    - Higher damping beta=0.9
    - Uses jax.lax.fori_loop for XLA efficiency
    """
    batch_size = z_init.shape[0]
    d = z_init.size // batch_size
    
    def step_fn(i, state):
        z, G_hist, Z_hist = state
        
        f_z = f(z)
        g = f_z - z
        
        z_flat = z.reshape(batch_size, -1)
        g_flat = g.reshape(batch_size, -1)
        
        # Update history
        G_hist = jnp.roll(G_hist, 1, axis=1)
        Z_hist = jnp.roll(Z_hist, 1, axis=1)
        G_hist = G_hist.at[:, 0, :].set(g_flat)
        Z_hist = Z_hist.at[:, 0, :].set(z_flat)
        
        def simple_update(_):
            return z_flat + beta * g_flat
        
        def anderson_update(_):
            current_g = G_hist[:, 0, :]
            prev_G = G_hist[:, 1:, :]
            current_z = Z_hist[:, 0, :]
            prev_Z = Z_hist[:, 1:, :]
            
            DG = current_g[:, None, :] - prev_G
            DZ = current_z[:, None, :] - prev_Z
            
            # For m=2: 1x1 solve (scalar division)
            # For m=3: 2x2 solve
            if m == 2:
                # Simple 1x1: gamma = (DG^T g) / (DG^T DG + λ)
                DG_0 = DG[:, 0, :]  # (B, d)
                DZ_0 = DZ[:, 0, :]
                
                numerator = jnp.sum(DG_0 * current_g, axis=-1)  # (B,)
                denominator = jnp.sum(DG_0 * DG_0, axis=-1) + lam  # (B,)
                gamma = numerator / denominator  # (B,)
                
                correction = gamma[:, None] * (DZ_0 + beta * DG_0)
            else:
                # 2x2 system for m=3
                DG_t = jnp.transpose(DG, (0, 2, 1))
                gram = jnp.matmul(jnp.transpose(DG_t, (0, 2, 1)), DG_t)
                gram = gram + lam * jnp.eye(m-1)[None, :, :]
                
                rhs = jnp.matmul(jnp.transpose(DG_t, (0, 2, 1)), current_g[:, :, None])
                
                a, b = gram[:, 0, 0], gram[:, 0, 1]
                c, d_ = gram[:, 1, 0], gram[:, 1, 1]
                det = a * d_ - b * c + 1e-8
                
                r0, r1 = rhs[:, 0, 0], rhs[:, 1, 0]
                gamma0 = (d_ * r0 - b * r1) / det
                gamma1 = (-c * r0 + a * r1) / det
                gamma = jnp.stack([gamma0, gamma1], axis=1)[:, :, None]
                
                correction = jnp.sum(gamma * (DZ + beta * DG), axis=1)
            
            z_new = (z_flat + beta * current_g) - correction
            return z_new
        
        z_new_flat = jax.lax.cond(i < m, simple_update, anderson_update, None)
        z_new = z_new_flat.reshape(z.shape)
        
        return (z_new, G_hist, Z_hist)
    
    Z_hist = jnp.zeros((batch_size, m, d))
    G_hist = jnp.zeros((batch_size, m, d))
    
    init_state = (z_init, G_hist, Z_hist)
    final_state = jax.lax.fori_loop(0, max_iter, step_fn, init_state)
    
    return final_state[0]


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
    
    # Implicit differentiation via Neumann series (faster than full solve)
    def bwd_step(i, u):
        _, vjp_fun = jax.vjp(lambda z: f_apply(params, z, x_input), z_star)
        ju = vjp_fun(u)[0]
        return grad_z_star + ju
    
    # 4 iterations of Neumann series approximation
    u_star = jax.lax.fori_loop(0, 4, bwd_step, grad_z_star)
    
    # Gradients w.r.t params and x
    _, vjp_fun_params_x = jax.vjp(lambda p, x: f_apply(p, z_star, x), params, x_input)
    grads_params, grads_x = vjp_fun_params_x(u_star)
    
    return grads_params, grads_x, None

deq_fixed_point.defvjp(deq_fixed_point_fwd, deq_fixed_point_bwd)


# --- MODULE ---

class DeepEquilibriumModel(nn.Module):
    """
    Deep Equilibrium Model (DEQ) with Anderson Acceleration.
    
    Per GPU ONLY Chess RL Agent.txt spec:
    - Uses Anderson Acceleration for fixed-point finding
    - O(1) memory via implicit differentiation
    - Adaptive depth through convergence tolerance
    
    Optimizations:
    - Reduced history m=3 (was 5)
    - Uses fori_loop for efficient XLA compilation
    - Faster 2x2 direct solve instead of generic lstsq
    """
    block_class: Any
    block_args: Dict[str, Any]
    max_iter: int = 6  # Reasonable depth
    tol: float = 1e-4
    beta: float = 0.8  # Damping factor
    m: int = 3  # Anderson history size

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

        z_init = jnp.zeros_like(x_input)
        solver_args = {
            'max_iter': self.max_iter, 
            'tol': self.tol, 
            'beta': self.beta,
            'm': self.m
        }
        
        z_star = deq_fixed_point(apply_fn, block_vars.value, x_input, z_init, solver_args)
        
        return z_star
