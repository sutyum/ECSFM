import jax
import jax.numpy as jnp
import equinox as eqx

class SignalEncoder(eqx.Module):
    """
    1D Convolutional network to encode arbitrary time-series signals (e.g. E(t))
    into a fixed-size latent representation.
    """
    layers: list
    input_channels: int = eqx.field(static=True)
    
    def __init__(self, cond_dim: int, key: jax.random.PRNGKey, input_channels: int = 1):
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.input_channels = int(input_channels)
        
        # We expect input of shape (channels, seq_len)
        self.layers = [
            eqx.nn.Conv1d(self.input_channels, 16, kernel_size=5, stride=2, padding=2, key=k1),
            jax.nn.gelu,
            eqx.nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, key=k2),
            jax.nn.gelu,
            eqx.nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, key=k3),
            jax.nn.gelu,
            # Adaptive pooling to sequence length of 1 to squash time dimension
            eqx.nn.AdaptiveAvgPool1d(1),
            # Flatten is conceptually: x.reshape(-1)
            eqx.nn.Linear(64, cond_dim, key=k4)
        ]
        
    def __call__(self, x: jax.Array) -> jax.Array:
        # Input x is expected to be shape (seq_len,) or (channels, seq_len).
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim != 2:
            raise ValueError(f"Signal input must be 1D or 2D, got shape {x.shape}")

        if x.shape[0] != self.input_channels:
            raise ValueError(
                f"Signal channels mismatch: expected {self.input_channels}, got {x.shape[0]}"
            )

        conv_dtype = self.layers[0].weight.dtype
        x = x.astype(conv_dtype)

        for layer in self.layers:
            # Special handling to flatten before linear layer
            if isinstance(layer, eqx.nn.Linear):
                x = x.reshape(-1)
            x = layer(x)
        return x

class VectorFieldNet(eqx.Module):
    """
    Neural network parameterizing the vector field v_theta(t, x, E).
    This takes continuous time `t` (integration time in flow matching, t in [0, 1]),
    the state `x` (e.g., flattened concentrations), and the full time-series signal `E(t)`.
    """
    signal_encoder: SignalEncoder
    mlp: eqx.nn.MLP
    dropout: eqx.nn.Dropout
    time_emb_dim: int = eqx.field(static=True)
    signal_channels: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        depth: int,
        cond_dim: int,
        phys_dim: int,
        key: jax.random.PRNGKey,
        signal_channels: int = 1,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            state_dim: Dimension of the state vector (e.g. C_ox, C_red, I_hist)
            hidden_size: Size of mlp hidden layers
            depth: Number of mlp hidden layers
            cond_dim: Size of the encoded latent vector from the signal E(t)
            phys_dim: Size of the explicit physical parameter vector p
            key: PRNG key for initialization
            dropout_rate: Dropout probability applied before the MLP
        """
        k1, k2 = jax.random.split(key)
        self.time_emb_dim = 32
        self.signal_channels = int(signal_channels)

        self.signal_encoder = SignalEncoder(
            cond_dim=cond_dim,
            key=k1,
            input_channels=self.signal_channels,
        )

        self.mlp = eqx.nn.MLP(
            in_size=state_dim + self.time_emb_dim + cond_dim + phys_dim,
            out_size=state_dim,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=k2
        )

        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, t: float, x: jax.Array, E: jax.Array, p: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """
        Predicts the vector field at time t, state x, conditioned on signal E and params p.
        Args:
            t: Scalar time
            x: State array of shape (state_dim,)
            E: Experimental signal array of shape (seq_len,)
            p: Physical parameter array of shape (phys_dim,)
            key: PRNG key for dropout (required during training, ignored in inference mode)
        """
        # Ensure t is a 1D array of shape (1,)
        t_arr = jnp.atleast_1d(t)

        # Sinusoidal time embedding
        half_dim = self.time_emb_dim // 2
        emb_scale = jnp.exp(jnp.arange(half_dim) * -(jnp.log(10000.0) / max(1, half_dim - 1)))
        emb = t_arr * emb_scale
        t_emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])

        # Process the time-series boundary signal E(t) into a fixed-size condition
        c = self.signal_encoder(E)

        # Concatenate time, state, compressed signal condition, and physical parameters
        txcp = jnp.concatenate([t_emb, x, c, p])

        # Apply dropout before MLP
        txcp = self.dropout(txcp, key=key)

        # Predict velocity
        return self.mlp(txcp)
