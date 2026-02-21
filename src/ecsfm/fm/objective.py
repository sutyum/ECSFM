import jax
import jax.numpy as jnp
import equinox as eqx

def flow_matching_loss(model, x1: jax.Array, x0: jax.Array, E: jax.Array, p: jax.Array, key: jax.random.PRNGKey):
    """
    Computes the Optimal Transport Conditional Flow Matching (OT-CFM) loss.
    
    In the context of the electrochemical simulator, we can map:
    - x0: A base distribution (e.g., standard normal noise, or the initial t=0 state)
    - x1: The target data distribution (e.g., the simulation outcome at t_final)
    - E: Conditioning time-series signal vector (e.g., applied voltage E(t))
    - p: Conditioning physical scalar parameters
    
    The OT path is: x_t = (1 - t) * x0 + t * x1
    The target vector field is: u_t = x1 - x0
    
    We train the model vector field v_theta(t, x_t, E, p) to match u_t.
    
    Args:
        model: The vector field neural network v_theta(t, x, E, p)
        x1: A batch of data samples, shape (batch_size, state_dim)
        x0: A batch of base samples, shape (batch_size, state_dim)
        E: A batch of conditioning signals, shape (batch_size, seq_len)
        p: A batch of physical parameters, shape (batch_size, phys_dim)
        key: PRNG key for sampling t
        
    Returns:
        Scalar loss value.
    """
    # Ensure they have the same shape
    batch_size = x1.shape[0]
    assert x1.shape == x0.shape, "x1 and x0 must have the same shape"
    assert x1.shape[0] == E.shape[0], "E must have the same batch size as x1 and x0"
    assert x1.shape[0] == p.shape[0], "p must have the same batch size as x1 and x0"
    
    # Sample a random time t for each item in the batch
    # t ~ U(0, 1)
    t = jax.random.uniform(key, shape=(batch_size, 1))
    
    # Construct the interpolated state x_t
    x_t = (1.0 - t) * x0 + t * x1
    
    # The true vector field guiding the flow from x0 to x1
    u_t = x1 - x0
    
    # Predict the vector field using the model
    # We vmap over the batch dimension
    vmap_model = jax.vmap(model)
    v_pred = vmap_model(t, x_t, E, p)
    
    # Compute the mean squared error
    loss = jnp.mean((v_pred - u_t) ** 2)
    return loss
