import os
import json
import subprocess
import itertools
from pathlib import Path

def tune():
    """
    Kicks off isolated training tuning runs using JSON configs 
    and saves the outputs individually.
    """
    learning_rates = [1e-3, 5e-4]
    depths = [3, 4]
    hidden_sizes = [64, 128]
    
    # Generate combinatorial configs
    configs = list(itertools.product(learning_rates, depths, hidden_sizes))
    print(f"Starting Grid Search: Found {len(configs)} configurations to tune.")
    
    # Store configs in a temporary directory
    tune_dir = Path("tune_configs")
    tune_dir.mkdir(exist_ok=True)
    
    for i, (lr, depth, hs) in enumerate(configs):
        print(f"\n--- Run {i+1}/{len(configs)}: LR={lr}, Depth={depth}, Hidden={hs} ---")
        
        config_path = tune_dir / f"config_run_{i}.json"
        
        # We specify new_run explicitly to ignore any overlapping checkpoints
        with open(config_path, "w") as f:
            json.dump({
                "lr": lr,
                "depth": depth,
                "hidden_size": hs,
                "epochs": 100, # A short test horizon to gauge fast convergence
                "n_samples": 100, 
                "batch_size": 64,
                "new_run": True, 
                "val_split": 0.2
            }, f, indent=4)
            
        print(f"Generated physical hyperparams at {config_path}")
        
        try:
            # We must use `python` as the entrypoint here since uv permission problems 
            # prevent nested uv execution dynamically depending on SIP bounds
            subprocess.run(
                ["python", "src/ecsfm/fm/train.py", "--config", str(config_path)],
                check=True,
                env={**os.environ, "PYTHONPATH": "src"} 
            )
            
            # Since train.py overwrites surrogate_model.eqx and training_history.json
            # Let's cleanly stash them 
            model_dest = tune_dir / f"model_run_{i}.eqx"
            hist_dest = tune_dir / f"history_run_{i}.json"
            
            if os.path.exists("surrogate_model.eqx"):
                os.rename("surrogate_model.eqx", model_dest)
            if os.path.exists("training_history.json"):
                os.rename("training_history.json", hist_dest)
                
            print(f"Successfully stashed artifacts for Run {i+1} in {tune_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"Tuning run failed with error: {e}")
            
    # Evaluation Phase: Post-process to find the lowest validation loss
    print("\n--- Hyperparameter Evaluation ---")
    best_loss = float('inf')
    best_config = None
    best_run = -1
    
    for i in range(len(configs)):
        hist_path = tune_dir / f"history_run_{i}.json"
        config_path = tune_dir / f"config_run_{i}.json"
        
        if not hist_path.exists() or not config_path.exists():
            continue
            
        with open(hist_path, "r") as f:
            hist_data = json.load(f)
            val_history = hist_data.get("history", {}).get("val", [])
            
        with open(config_path, "r") as f:
            config_data = json.load(f)
            
        if val_history:
            # val_history is a list of [epoch, loss]
            # Get the minimum validation loss across the run
            min_val_loss = min([item[1] for item in val_history])
            print(f"Run {i}: Min Val Loss = {min_val_loss:.5f} | LR={config_data['lr']}, Depth={config_data['depth']}, Hidden={config_data['hidden_size']}")
            
            if min_val_loss < best_loss:
                best_loss = min_val_loss
                best_config = config_data
                best_run = i
    
    if best_config:
        print(f"\nWINNER (Run {best_run}): Min Val Loss = {best_loss:.5f}")
        # The user requested to overwrite defaults, let's reset epochs to a larger training horizon
        # Keep the winning LR, Depth, and Hidden Size.
        best_config["epochs"] = 1000 
        best_config["n_samples"] = 500
        best_config["new_run"] = False # Next time they run train.py it should naturally resume or start fresh normally
        
        final_config_path = Path("config.json")
        with open(final_config_path, "w") as f:
            json.dump(best_config, f, indent=4)
            
        print(f"Successfully wrote the winning hyperparameters to {final_config_path} !!")
    else:
        print("\nNo successful tuning runs found to evaluate.")
    tune()
