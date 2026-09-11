import h5py
import json

def inspect_h5_structure(file_path):
    # This function will be called for every single item inside the .h5 file
    def print_node_info(name, node):
        if isinstance(node, h5py.Dataset):
            # If it is an actual tensor/array containing weights
            print(f"  [Dataset] {name} | Shape: {node.shape} | Type: {node.dtype}")
        elif isinstance(node, h5py.Group):
            # If it is just a folder/container for layers
            print(f"[Group] {name}")

    print(f"Inspecting structure of: {file_path}\n" + "="*50)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # visititems recursively traverses the entire HDF5 file tree
            f.visititems(print_node_info)

            # Keras saves the full architecture JSON in the root attributes
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'])
                print(json.dumps(config, indent=2))
            else:
                print("Keys in f.attrs:", list(f.attrs.keys()))
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'.")

# Replace with the actual path to the downloaded model
inspect_h5_structure('NeuralForceFunctional.keras')
