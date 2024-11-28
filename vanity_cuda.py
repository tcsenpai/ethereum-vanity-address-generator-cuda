import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import json
import os
from dotenv import load_dotenv
from libs.cudacode import CUDA_CRYPTO_CODE

# Load environment variables
load_dotenv()
PREFIX = os.getenv("PREFIX", "dEAD000000000000000042069420694206942069").lower()
BATCH_SIZE = 2**16  # 65536 addresses per batch


class EthereumVanityMiner:
    def __init__(self):
        # Initialize CUDA
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()

        # Compile CUDA module
        self.mod = SourceModule(CUDA_CRYPTO_CODE, no_extern_c=True)
        self.kernel = self.mod.get_function("generate_and_check")

        # Prepare memory buffers
        self.gpu_private_keys = cuda.mem_alloc(BATCH_SIZE * 32)
        self.gpu_addresses = cuda.mem_alloc(BATCH_SIZE * 20)
        self.gpu_match_lengths = cuda.mem_alloc(BATCH_SIZE * 4)

        # Prepare target prefix
        prefix = PREFIX[2:] if PREFIX.startswith("0x") else PREFIX
        # Debug print
        print(f"Converting prefix: {prefix}")
        
        # Convert each pair of hex chars to a byte
        self.prefix_bytes = bytearray()
        for i in range(0, len(prefix), 2):
            hex_pair = prefix[i:i+2]
            byte_val = int(hex_pair, 16)
            self.prefix_bytes.append(byte_val)
        
        # Debug print
        print(f"Prefix bytes: {[hex(b) for b in self.prefix_bytes]}")
        
        self.gpu_target = cuda.mem_alloc(len(self.prefix_bytes))
        cuda.memcpy_htod(self.gpu_target, self.prefix_bytes)

        # Host buffers
        self.host_match_lengths = np.zeros(BATCH_SIZE, dtype=np.int32)
        self.host_private_key = np.zeros(32, dtype=np.uint8)
        self.host_address = np.zeros(20, dtype=np.uint8)

    def cleanup(self):
        """Clean up GPU resources"""
        try:
            self.gpu_private_keys.free()
            self.gpu_addresses.free()
            self.gpu_match_lengths.free()
            self.gpu_target.free()
        finally:
            self.context.pop()
            self.context.detach()

    def save_match(
        self, address_hex, private_key_hex, match_length, raw_address, raw_private_key
    ):
        """Save match to JSON file"""
        try:
            matches = []
            if os.path.exists("cuda_matches.json"):
                with open("cuda_matches.json", "r") as f:
                    matches = json.load(f)

            match_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "address": f"0x{address_hex}",
                "private_key": private_key_hex,
                "match_length": int(match_length),
                "raw_address": raw_address,
                "raw_private_key": raw_private_key,
            }
            matches.append(match_data)

            with open("cuda_matches.json", "w") as f:
                json.dump(matches, f, indent=2)
        except Exception as e:
            print(f"\nWarning: Failed to save match to file: {e}")

    def check_batch(self, seed):
        """Run one batch of address generation and checking"""
        block = (256, 1, 1)
        grid = ((BATCH_SIZE + block[0] - 1) // block[0], 1)
        
        self.kernel(
            self.gpu_private_keys,
            self.gpu_addresses,
            self.gpu_match_lengths,
            self.gpu_target,
            np.int32(len(self.prefix_bytes)),
            np.uint32(seed),
            block=block,
            grid=grid
        )
        
        # Get results
        cuda.memcpy_dtoh(self.host_match_lengths, self.gpu_match_lengths)
        best_idx = np.argmax(self.host_match_lengths)
        match_length = self.host_match_lengths[best_idx]
        
        if match_length > 0:
            temp_private_key = np.zeros(32, dtype=np.uint8)
            temp_address = np.zeros(20, dtype=np.uint8)
            
            cuda.memcpy_dtoh(temp_private_key, self.gpu_private_keys)
            cuda.memcpy_dtoh(temp_address, self.gpu_addresses)
            
            address_hex = ''.join(format(x, '02x') for x in temp_address)
            private_key_hex = ''.join(format(x, '02x') for x in temp_private_key)
            
            # Verify the match
            target = PREFIX[2:] if PREFIX.startswith('0x') else PREFIX
            actual_match = 0
            for i, (t, a) in enumerate(zip(target, address_hex)):
                if t.lower() != a.lower():
                    break
                actual_match += 1
            
            if actual_match > 0:
                return {
                    'address': address_hex,
                    'private_key': private_key_hex,
                    'match_length': actual_match,
                    'raw_address': temp_address.tobytes().hex(),
                    'raw_private_key': temp_private_key.tobytes().hex()
                }
        return None

    def mine(self, target_score=None):
        """Main mining loop"""
        try:
            best_match = {"match_length": 0}
            start_time = time.time()
            last_status_time = start_time
            addresses_checked = 0

            while True:
                seed = np.random.randint(0, 2**32, dtype=np.uint32)
                result = self.check_batch(seed)

                addresses_checked += BATCH_SIZE
                current_time = time.time()

                # Print status every 5 seconds
                if current_time - last_status_time >= 5:
                    elapsed_time = current_time - start_time
                    rate = addresses_checked / elapsed_time
                    print(
                        f"\rChecked {addresses_checked:,} addresses ({rate:,.0f}/s) - Best match: {best_match['match_length']} chars",
                        end="",
                    )
                    last_status_time = current_time

                if result and result["match_length"] > best_match["match_length"]:
                    best_match = result
                    print(f"\nNew best match ({result['match_length']} chars):")
                    print(f"Address: 0x{result['address']}")
                    print(f"Private key: {result['private_key']}")

                    self.save_match(
                        result["address"],
                        result["private_key"],
                        result["match_length"],
                        result["raw_address"],
                        result["raw_private_key"],
                    )

                    if target_score and result["match_length"] >= target_score:
                        return result

        except KeyboardInterrupt:
            print("\nMining interrupted by user")
            return best_match
        finally:
            self.cleanup()


if __name__ == "__main__":
    try:
        miner = EthereumVanityMiner()
        print(f"Mining with {cuda.Device.count()} GPU(s)")
        print(f"Target prefix: {PREFIX}")
        print(f"Batch size: {BATCH_SIZE}")

        result = miner.mine(target_score=len(PREFIX))

        if result:
            print("\nFinal result:")
            print(f"Address: 0x{result['address']}")
            print(f"Private key: {result['private_key']}")
            print(f"Match length: {result['match_length']}")
    except Exception as e:
        print(f"Error: {e}")
