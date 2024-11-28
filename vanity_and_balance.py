import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from eth_keys import keys
from eth_utils import to_checksum_address
import time
import dotenv
import os
from web3 import Web3, AsyncWeb3
import asyncio
from collections import deque
import aiohttp
import concurrent.futures
from threading import Lock
import json

cuda.init()

dotenv.load_dotenv()

# Add near the top after imports
RPC_URLS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
    "https://ethereum.publicnode.com",
    "https://1rpc.io/eth",
]

# Add new configuration from .env
CHECK_BALANCES = os.getenv("CHECK_BALANCES", "false").lower() == "true"
if not CHECK_BALANCES:
    print("🚫 Balance checking is disabled.")
BALANCE_BATCH_SIZE = int(os.getenv("BALANCE_BATCH_SIZE", "100"))
SYNC_MODE = os.getenv("SYNC_MODE", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))  # Number of addresses to check at once
total_balance_checks = 0

# Create a round-robin RPC selector
rpc_index = 0

# Add near the top after other global variables
last_balance_check = {"address": None, "balance": None, "rpc": None}
pending_tasks = []
MAX_PENDING_TASKS = 10  # Adjust based on your needs

# Add near the top with other globals
FOUND_FILE = "found_addresses.txt"
BEST_MATCHES_FILE = "best_matches.json"


def get_next_web3():
    global rpc_index
    web3 = Web3(Web3.HTTPProvider(RPC_URLS[rpc_index]))
    rpc_index = (rpc_index + 1) % len(RPC_URLS)
    return web3


# Create a queue for addresses to check
address_queue = deque()

# Add to imports and configuration
print_lock = Lock()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
pending_checks = []

# Add this near the top of the file, after imports
cuda_code = """
#include <curand_kernel.h>

__device__ __forceinline__ void generate_private_key(unsigned char *out, const int idx, curandState *state) {
    // Generate 32 bytes (256 bits) for the private key using vectorized operations
    uint4 *out_vec = (uint4*)&out[idx * 32];
    
    // Generate 4 random 32-bit values using CUDA's optimized RNG
    uint4 rand_val;
    rand_val.x = curand(state);
    rand_val.y = curand(state);
    rand_val.z = curand(state);
    rand_val.w = curand(state);
    
    // Store using vector operation (more efficient than byte-by-byte)
    *out_vec = rand_val;
}

extern "C" __global__ void generate_random(unsigned char *out, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Initialize CUDA RNG state
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);
    
    // Generate private key using vectorized operations
    generate_private_key(out, idx, &state);
}
"""


def check_single_balance(address, private_key):
    """Non-blocking balance check with RPC failover"""
    for _ in range(len(RPC_URLS)):  # Try each RPC once
        try:
            web3 = get_next_web3()
            balance = web3.eth.get_balance(address)
            # Update last balance check info
            global last_balance_check
            last_balance_check = {
                "address": address,
                "balance": Web3.from_wei(balance, "ether"),
                "rpc": web3.provider.endpoint_uri,
            }
            if balance > 0:
                with print_lock:
                    print(f"\n{'='*50}")
                    print(f"🔥 Found address with balance!")
                    print(f"Address: {address}")
                    print(f"Balance: {Web3.from_wei(balance, 'ether')} ETH")
                    print(f"Private key: {private_key}")
                    print(f"{'='*50}\n")
                    with open(FOUND_FILE, "a") as f:
                        f.write(f"Address: {address}\nPrivate Key: {private_key}\nBalance: {Web3.from_wei(balance, 'ether')} ETH\n{'='*50}\n")
            return balance
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                with print_lock:
                    print(
                        f"⚠️  Rate limit hit on {web3.provider.endpoint_uri}: {str(e)}"
                    )
                time.sleep(1)  # Back off a bit
            else:
                with print_lock:
                    print(
                        f"⚠️  Error on {web3.provider.endpoint_uri}, trying next RPC: {str(e)}"
                    )
            continue

    with print_lock:
        print(f"❌ All RPCs failed to check balance for {address}")
    return None


async def generate_vanity_address(prefix, num_attempts=0):
    mod = SourceModule(
        cuda_code, no_extern_c=True, include_dirs=["/usr/local/cuda/include"]
    )
    generate_random = mod.get_function("generate_random")

    # Use up to 4GB of GPU memory
    max_memory = 4 * 1024 * 1024 * 1024  # 4GB in bytes
    max_batch = max_memory // 32  # Each key is 32 bytes
    current_batch = min(max_batch, 4000000)  # Start with 4 million addresses

    # Create pinned memory directly instead of converting
    private_keys = cuda.pagelocked_empty((current_batch, 32), dtype=np.uint8)
    try:
        gpu_private_keys = cuda.mem_alloc(private_keys.nbytes)
    except cuda.LogicError as e:
        print(
            f"⚠️ GPU memory allocation failed with batch size {current_batch}. Trying smaller batch..."
        )
        current_batch = current_batch // 2
        private_keys = cuda.pagelocked_empty((current_batch, 32), dtype=np.uint8)
        gpu_private_keys = cuda.mem_alloc(private_keys.nbytes)

    # Optimize thread configuration for RTX 4060
    block_size = 1024  # Maximum threads per block
    grid_size = (current_batch + block_size - 1) // block_size

    print(f"\n🔍 Starting search for prefix: {prefix}")
    print(
        f"💾 Batch size: {current_batch:,} addresses ({private_keys.nbytes / 1024 / 1024:.1f} MB)"
    )
    print(f"🧮 Grid size: {grid_size} blocks × {block_size} threads")
    print(f"💡 This might take a while depending on prefix length...")

    start_time = time.time()
    total_attempts = 0
    prefixes_checked = 0
    best_match = {"address": None, "similarity": 0, "private_key": None}
    last_status_time = time.time()
    active_checks = 0

    address_batch = []
    private_key_batch = []

    while True:
        generate_random(
            gpu_private_keys,
            np.int32(current_batch),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        cuda.memcpy_dtoh(private_keys, gpu_private_keys)
        total_attempts += current_batch

        # Process addresses without waiting for balance checks
        for priv_key in private_keys:
            priv_key_hex = "".join(format(x, "02x") for x in priv_key)

            try:
                private_key = keys.PrivateKey(bytes.fromhex(priv_key_hex))
                public_key = private_key.public_key
                address = to_checksum_address(public_key.to_address())
                addr_without_prefix = address[2:].lower()

                # Check for exact match first (fast path)
                if addr_without_prefix.startswith(prefix.lower()):
                    # Found a match! Queue balance check if needed
                    if CHECK_BALANCES:
                        address_batch.append(address)
                        private_key_batch.append(priv_key_hex)
                    return private_key, address

                # Track best partial match
                similarity = calculate_similarity(addr_without_prefix, prefix.lower())
                if similarity > best_match["similarity"]:
                    best_match = {
                        "address": address,
                        "similarity": similarity,
                        "private_key": priv_key_hex,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    print(f"🎯 Best match so far: {best_match['address']} ({best_match['similarity']} chars)")
                    
                    # Save to JSON file
                    try:
                        # Load existing matches
                        matches = []
                        try:
                            with open(BEST_MATCHES_FILE, 'r') as f:
                                matches = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            matches = []
                        
                        # Add new match
                        matches.append(best_match)
                        
                        # Save updated matches
                        with open(BEST_MATCHES_FILE, 'w') as f:
                            json.dump(matches, f, indent=2)
                    except Exception as e:
                        print(f"❌ Error saving best match: {str(e)}")
                    # Immediately check for balance if best match without waiting
                    ethBalance = check_single_balance(address, priv_key_hex)
                    print(f"💰 Balance: {ethBalance} ETH")
                # Queue balance check if enabled (without waiting)
                if CHECK_BALANCES:
                    address_batch.append(address)
                    private_key_batch.append(priv_key_hex)
                    if len(address_batch) >= BALANCE_BATCH_SIZE:
                        # Create task without waiting
                        task = asyncio.create_task(
                            batch_check_balances(
                                address_batch.copy(), private_key_batch.copy()
                            )
                        )
                        pending_tasks.append(task)
                        active_checks += 1

                        # Process completed tasks without blocking
                        await process_pending_tasks()

                        address_batch.clear()
                        private_key_batch.clear()

            except Exception as e:
                print(f"❌ Error generating address: {str(e)}")
                continue

            # Status update every 10 seconds
            current_time = time.time()
            if current_time - last_status_time >= 10:
                elapsed_time = current_time - start_time
                speed = total_attempts / elapsed_time
                prefix_speed = prefixes_checked / elapsed_time
                print(f"\n{'='*30} Status Update {'='*30}")
                print(f"⏱️  Time elapsed: {elapsed_time:.1f}s")
                print(f"🔢 Attempts: {total_attempts:,}")
                print(f"🔍 Prefixes checked: {prefixes_checked:,}")
                print(f"⚡ Speed: {speed:,.2f} addr/s")
                print(f"🚀 Prefix check speed: {prefix_speed:,.2f} prefixes/s")
                if CHECK_BALANCES:
                    print(f"✓ Total balance checks: {total_balance_checks:,}")
                    if last_balance_check["address"]:
                        print(f"📊 Last check: {last_balance_check['address']} - {last_balance_check['balance']} ETH")
                print(f"🎯 Best match so far: {best_match['address']} ({best_match['similarity']} chars)")
                print(f"{'='*72}\n")
                last_status_time = current_time

            prefixes_checked += 1  # we checked a key

        # Process any remaining addresses in the batch
        if CHECK_BALANCES and address_batch:
            future = asyncio.create_task(
                batch_check_balances(address_batch.copy(), private_key_batch.copy())
            )
            active_checks += 1

        # Only break if num_attempts is positive
        if num_attempts > 0 and total_attempts >= num_attempts:
            print(f"\n⚠️ Reached maximum attempts: {num_attempts:,}")
            break

    # Process any remaining tasks before exiting
    if pending_tasks:
        await asyncio.gather(*pending_tasks)

    return None, None


def suggest_leet_alternatives(text):
    # Only include leet mappings that result in valid hex characters
    leet_map = {
        "a": "4",
        "e": "3",
        "i": "1",
        "o": "0",
        "s": "5",
        "b": "8",
        # Removed 't':'7' and 'g':'9' as they're not valid hex
    }
    alternatives = [text]

    # Generate variations
    for char, leet_char in leet_map.items():
        for existing in alternatives.copy():
            if char in existing.lower():
                alternatives.append(existing.lower().replace(char, leet_char))

    # Filter to ensure only valid hex strings are suggested
    valid_chars = set("0123456789abcdefABCDEF")
    valid_alternatives = [
        alt for alt in alternatives if all(c in valid_chars for c in alt)
    ]

    return list(set(valid_alternatives))[:5]  # Return up to 5 unique suggestions


def sanitize_prefix(prefix):
    # Remove '0x' if present
    prefix = prefix.replace("0x", "")

    # Check for valid hex characters
    valid_chars = set("0123456789abcdefABCDEF")
    if not all(c in valid_chars for c in prefix):
        invalid_chars = [c for c in prefix if c not in valid_chars]
        suggestions = suggest_leet_alternatives(prefix)
        raise ValueError(
            f"Invalid characters in prefix: {invalid_chars}\n"
            f"Prefix must only contain hex characters (0-9, a-f)\n"
            f"Try these leet alternatives: {', '.join(suggestions)}"
        )

    return prefix.lower()


def calculate_similarity(address, prefix):
    """Count matching characters from start of address with prefix"""
    for i, (a, p) in enumerate(zip(address, prefix)):
        if a != p:
            return i
    return len(prefix)


async def batch_check_balances(addresses, private_keys):
    """Check multiple balances in a single RPC call with failover"""
    global total_balance_checks
    
    for _ in range(len(RPC_URLS)):  # Try each RPC once
        try:
            web3 = get_next_web3()

            # Create batch payload
            payload = [
                {
                    "jsonrpc": "2.0",
                    "method": "eth_getBalance",
                    "params": [addr, "latest"],
                    "id": i,
                }
                for i, addr in enumerate(addresses)
            ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    web3.provider.endpoint_uri, json=payload
                ) as response:
                    results = await response.json()

            # Process results
            for result, address, private_key in zip(results, addresses, private_keys):
                if "result" in result:
                    total_balance_checks += 1
                    balance = int(result["result"], 16)  # Convert hex to int
                    # Update last balance check info
                    global last_balance_check
                    last_balance_check = {
                        "address": address,
                        "balance": Web3.from_wei(balance, "ether"),
                        "rpc": web3.provider.endpoint_uri,
                    }
                    # print(f"📊 Updated last_balance_check: {last_balance_check}")
                    if balance > 0:
                        with print_lock:
                            print(f"\n{'='*50}")
                            print(f"🔥 Found address with balance!")
                            print(f"Address: {address}")
                            print(f"Balance: {Web3.from_wei(balance, 'ether')} ETH")
                            print(f"Private key: {private_key}")
                            print(f"{'='*50}\n")
                            with open(FOUND_FILE, "a") as f:
                                f.write(f"Address: {address}\nPrivate Key: {private_key}\nBalance: {Web3.from_wei(balance, 'ether')} ETH\n{'='*50}\n")

            return results

        except Exception as e:
            if "429" in str(e):  # Rate limit error
                continue  # skip this rpc for now
            else:
                with print_lock:
                    print(
                        f"⚠️  Batch check error on {web3.provider.endpoint_uri}, trying next RPC: {str(e)}"
                    )
            continue

    with print_lock:
        print(
            f"❌ All RPCs failed for batch balance check of {len(addresses)} addresses"
        )
    return None


async def determine_optimal_batch_size(rpc_url):
    print("🔍 Testing RPC batch size limits...")
    optimal_size = 100  # Default fallback

    test_sizes = [2000, 1000, 500, 100]
    for batch_size in test_sizes:
        try:
            web3 = Web3(Web3.HTTPProvider(rpc_url))
            addresses = ["0x" + "0" * 40] * batch_size

            payload = [
                {
                    "jsonrpc": "2.0",
                    "method": "eth_getBalance",
                    "params": [addr, "latest"],
                    "id": i,
                }
                for i, addr in enumerate(addresses)
            ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    web3.provider.endpoint_uri, json=payload
                ) as response:
                    results = await response.json()
                    if isinstance(results, list) and len(results) == batch_size:
                        optimal_size = batch_size
                        print(f"✅ Found optimal batch size: {optimal_size}")
                        return optimal_size

        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {str(e)}")
            continue

    print(f"ℹ️ Using conservative batch size: {optimal_size}")
    return optimal_size


async def process_pending_tasks():
    """Process completed tasks and remove them from the pending list"""
    global pending_tasks

    # Only process completed tasks
    done_tasks = [task for task in pending_tasks if task.done()]
    for task in done_tasks:
        try:
            result = await task
            if result:
                # Process result if needed
                pass
        except Exception as e:
            print(f"❌ Task failed: {str(e)}")
        pending_tasks.remove(task)

    # If we have too many pending tasks, wait for some to complete
    if len(pending_tasks) >= MAX_PENDING_TASKS:
        if pending_tasks:
            await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)


async def main():
    prefix = os.getenv("PREFIX")
    if not prefix:
        prefix = input("Enter desired address prefix (without 0x): ")

    try:
        prefix = sanitize_prefix(prefix)
        private_key, address = await generate_vanity_address(prefix)

        # Cleanup thread pool
        executor.shutdown(wait=False)

        return private_key, address

    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    SUPPORTED_SIZES = []
    if CHECK_BALANCES:
        for url in RPC_URLS:
            SUPPORTED_SIZES.append(asyncio.run(determine_optimal_batch_size(url)))
        BALANCE_BATCH_SIZE = min(SUPPORTED_SIZES)
        print(f"🎯 Using safe batch size of {BALANCE_BATCH_SIZE} for all RPCs\n")
    else:
        print("🚫 Balance checking is disabled.")

    asyncio.run(main())
