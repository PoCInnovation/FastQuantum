import json
import os

class CircuitRegistry:
    """
    Manages the mapping between unique Circuit Names and integer IDs for the Embedding Layer.
    ID 0 is reserved for <UNK> (Unknown circuits).
    """
    def __init__(self, save_path="circuit_map.json"):
        self.save_path = save_path
        self.name_to_id = {"<UNK>": 0}
        self.next_id = 1
        
        # Load automatically if exists
        if os.path.exists(self.save_path):
            self.load()
    
    def get_id(self, circuit_name, create_if_missing=True):
        """
        Get the ID for a given circuit name.
        If create_if_missing is True, assigns a new ID for unknown names.
        If False, returns 0 (<UNK>) for unknown names.
        """
        if circuit_name not in self.name_to_id:
            if create_if_missing:
                self.name_to_id[circuit_name] = self.next_id
                self.next_id += 1
                self.save() # Auto-save on update
            else:
                return self.name_to_id["<UNK>"]
        return self.name_to_id[circuit_name]
    
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.name_to_id, f, indent=4)
            
    def load(self):
        with open(self.save_path, 'r') as f:
            self.name_to_id = json.load(f)
            # Update next_id to be max_id + 1
            if self.name_to_id:
                self.next_id = max(self.name_to_id.values()) + 1
            else:
                self.next_id = 1

    def __len__(self):
        return len(self.name_to_id)
