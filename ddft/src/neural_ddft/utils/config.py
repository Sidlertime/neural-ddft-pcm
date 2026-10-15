import argparse
import json
import os

def load_config(param_scheme, description="Neural DDFT Script"):
    """
    Parses CLI arguments and merges them with a JSON config file.
    CLI arguments always take priority over JSON.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    for name, (arg_type, default) in param_scheme.items():
        if arg_type not in [list, dict]:
            if arg_type is bool:
                parser.add_argument(f"--{name}", action="store_true")
            else:
                parser.add_argument(f"--{name}", type=arg_type)
            
    args = parser.parse_args()
    
    config_data = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            
    final_args = {}
    for name, (arg_type, default) in param_scheme.items():
        if arg_type in [list, dict]:
            # Bypass argparse entirely for complex nested structures
            final_args[name] = config_data.get(name, default)
        else:
            cli_val = getattr(args, name)
            if arg_type is bool:
                # If a flag is passed in CLI, it is True. Otherwise, check JSON, then default.
                if cli_val is True:
                    final_args[name] = True
                else:
                    final_args[name] = config_data.get(name, default)
            else:
                if cli_val is not None:
                    final_args[name] = cli_val
                else:
                    final_args[name] = config_data.get(name, default)
                
    print(f"Loaded Config:\n{json.dumps(final_args, indent=2)}\n")
    return final_args

def get_next_result_path(base_name):
    counter = 1
    while counter < 100:
        path = f"{base_name}_{counter:02d}"
        if not os.path.exists(path):
            return path
        counter += 1

def save_config(config_dict, save_dir):
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
