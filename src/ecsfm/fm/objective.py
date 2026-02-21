import jax
import jax.numpy as jnp
import equinox as eqx

def flow_matching_loss(model, x1: jax.Array, x0: jax.Array, key: jax.random.PRNGKey):
    """
    Computes the Optimal Transport Conditional Flow Matching (OT-CFM) loss.
    
    In the context of the electrochemical simulator, we can map:
    - x0: A base distribution (e.g., standard normal noise, or the initial t=0 state)
    - x1: The target data distribution (e.g., the simulation outcome at t_final)
    
    The OT path is: x_t = (1 - t) * x0 + t * x1
    The target vector field is: u_t = x1 - x0
    
    We train the model vector field v_theta(t, x_t) to match u_t.
    
    Args:
        model: The vector field neural network v_theta(t, x)
        x1: A batch of data samples, shape (batch_size, state_dim)
        x0: A batch of base samples, shape (batch_size, state_dim)
        key: PRNG key for sampling t
        
    Returns:
        Scalar loss value.
    """
    # Ensure they have the same shape
    batch_size = x1.shape[0]
    assert x1.shape == x0.shape, "x1 and x0 must have the same shape"
    
    # Sample a random time t for each item in the batch
    # t ~ U(0, 1)
    t = jax.random.uniform(key, shape=(batch_size, 1))
    
    # Construct the interpolated state x_t
    x_t = (1.0 - t) * x0 + t * x1
    
    # The true vector field guiding the flow from x0 to x1
    u_t = x1 - x0
    
    # Predict the vector field using the model
    # We vmap over the batch dimension
    # model(t, x) expects scalar t and vector x, so we vmap it
    vmap_model = jax.vmap(model)
    v_pred = vmap_model(t, x_t)
    
    # Compute the mean squared error
    loss = jnp.mean((v_pred - u_t) ** 2)
    return loss
