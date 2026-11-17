"""Print the structure of an HDF5 file (datasets, shapes, Keras model config), e.g. an MD frame or a .keras model.

Usage: python scripts/tools/h5_inspect.py <file.h5> [<file2.h5> ...]
"""
import sys
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        inspect_h5_structure(path)
