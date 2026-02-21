import jax
import jax.numpy as jnp
import equinox as eqx

class VectorFieldNet(eqx.Module):
    """
    Neural network parameterizing the vector field v_theta(t, x).
    This takes continuous time `t` (integration time in flow matching, t in [0, 1])
    and the state `x` (e.g., flattened concentrations), and predicts the velocity.
    """
    mlp: eqx.nn.MLP
    time_emb_dim: int = eqx.field(static=True)
    
    def __init__(self, state_dim: int, hidden_size: int, depth: int, key: jax.random.PRNGKey):
        """
        Args:
            state_dim: Dimension of the state vector (e.g. 2 * nx for (C_ox, C_red))
            hidden_size: Size of hidden layers
            depth: Number of hidden layers
            key: PRNG key for initialization
        """
        self.time_emb_dim = 32
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + self.time_emb_dim,
            out_size=state_dim,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=key
        )
        
    def __call__(self, t: float, x: jax.Array) -> jax.Array:
        """
        Predicts the vector field at time t and state x.
        """
        # Ensure t is a 1D array of shape (1,)
        t_arr = jnp.atleast_1d(t)
        
        # Sinusoidal time embedding (crucial for Flow Matching & Score ODEs)
        half_dim = self.time_emb_dim // 2
        emb_scale = jnp.exp(jnp.arange(half_dim) * -(jnp.log(10000.0) / max(1, half_dim - 1)))
        emb = t_arr * emb_scale
        t_emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
        
        # Concatenate time and state
        tx = jnp.concatenate([t_emb, x])
        
        # Predict velocity
        return self.mlp(tx)

