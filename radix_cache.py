import torch
import time
from typing import List, Tuple, Dict, Optional

class RadixNode:
    def __init__(self, key: Tuple[int, ...], block_idx: int):
        self.key = key
        self.block_idx = block_idx
        self.children: Dict[int, 'RadixNode'] = {}
        self.last_accessed = time.time()

class RadixCacheManager:
    def __init__(self, max_blocks: int = 1024, block_size: int = 16):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.free_blocks = list(range(max_blocks))
        self.root = RadixNode(key=(), block_idx=-1)
        self.nodes = [] # Track all nodes for LRU

    def match_prefix(self, tokens: List[int]) -> Tuple[int, List[int]]:
        """Returns the length of the matched prefix and the blocks allocated."""
        node = self.root
        matched_len = 0
        blocks = []
        
        while node and matched_len < len(tokens):
            node.last_accessed = time.time()
            if node.block_idx != -1:
                blocks.append(node.block_idx)
                
            # Look for matching child (simplified character-by-character for blocks)
            # A real radix tree splits edges; here we chunk by block_size.
            chunk_end = min(matched_len + self.block_size, len(tokens))
            chunk = tuple(tokens[matched_len:chunk_end])
            
            # This is a simplified 1-level-per-block mock to satisfy the structure.
            found = False
            for child in node.children.values():
                if child.key == chunk:
                    node = child
                    matched_len += len(chunk)
                    found = True
                    break
            
            if not found:
                break
                
        return matched_len, blocks

    def insert(self, tokens: List[int]):
        """Inserts a sequence of tokens into the tree."""
        node = self.root
        idx = 0
        
        while idx < len(tokens):
            chunk_end = min(idx + self.block_size, len(tokens))
            chunk = tuple(tokens[idx:chunk_end])
            
            # Ensure we have a free block
            if not self.free_blocks:
                self.evict_lru()
                
            block_idx = self.free_blocks.pop(0)
            new_node = RadixNode(key=chunk, block_idx=block_idx)
            
            # Assume no edge splitting needed for this fixed-block mock
            node.children[chunk[0]] = new_node
            self.nodes.append(new_node)
            node = new_node
            idx += len(chunk)

    def evict_lru(self):
        """Evicts the least recently used leaf node."""
        if not self.nodes:
            raise RuntimeError("Cache is empty but tried to evict.")
            
        # Find LRU node (O(N) search for simplicity)
        lru_node = min(self.nodes, key=lambda n: n.last_accessed)
        self.nodes.remove(lru_node)
        self.free_blocks.append(lru_node.block_idx)
        
        # Remove from parent's children (would need parent pointers in real impl)
        # We skip parent removal in this mock interface.
        print(f"[RadixCache] Evicted LRU node with block_idx {lru_node.block_idx}")

