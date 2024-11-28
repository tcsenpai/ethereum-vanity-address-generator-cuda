# Ethereum Vanity Address Generator with CUDA

A high-performance Ethereum vanity address generator that uses CUDA GPU acceleration to quickly generate Ethereum addresses matching a desired prefix pattern. It includes optional balance checking functionality across multiple RPC endpoints and tracks best partial matches.

## Features

- 🚀 CUDA GPU acceleration for fast address generation
- 🔍 Configurable prefix matching with support for hex characters
- 💰 Optional balance checking across multiple RPC endpoints
- ⚡ Batch processing for efficient balance checks
- 🔄 Automatic RPC failover and rate limit handling
- 📊 Real-time status updates and progress tracking
- 💾 Automatic saving of matches and addresses with balances
- 🎯 Tracks best partial matches and saves them to JSON
- 🔐 Secure private key generation using CUDA's RNG

## Requirements

- Python 3.7+
- CUDA-capable NVIDIA GPU
- PyCUDA
- Web3.py
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tcsenpai/eth-vanity-address-generator-cuda.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure settings in `.env` file:

   ```bash
   # Enable balance checking (true/false)
   CHECK_BALANCES=true

   # RPC URLs for redundancy
   RPC_URLS=[
       "https://eth.llamarpc.com",
       "https://rpc.ankr.com/eth",
       "https://ethereum.publicnode.com",
       "https://1rpc.io/eth"
   ]

   # How many addresses to check in each batch
   BALANCE_BATCH_SIZE=100

   # Desired address prefix
   PREFIX=dEAD000000000000000042069420694206942069

   # Set to true for synchronous balance checking
   SYNC_MODE=false

   # CUDA batch size for address generation
   BATCH_SIZE=500
   ```

## Usage

You can run either the pure CUDA version or the version with balance checking:

1. CUDA-only version:

   ```bash
   ./find_address_fullcuda.sh
   ```

2. CUDA with balance checking:
   ```bash
   ./find_address_and_check_balance.sh
   ```

The script will begin generating addresses and checking balances if enabled. Status updates are printed every 10 seconds (5s for the full cuda version without balance checking) and show:

- Time elapsed
- Total attempts
- Generation speed
- Prefix check speed
- Balance check status
- Best match found so far

When a matching address is found, it will be displayed along with its private key.

## Output Files

- `found_addresses.txt`: Contains addresses found with balances
- `best_matches.json`: Tracks the best partial matches found
- `cuda_matches.json`: Records all matches found by the CUDA miner

## Configuration

The following settings can be configured in the `.env` file:

- `CHECK_BALANCES`: Enable/disable balance checking (only works with balance checking version)
- `RPC_URLS`: List of Ethereum RPC endpoints for redundancy (only works with balance checking version)
- `BALANCE_BATCH_SIZE`: Number of addresses to check in each batch (only works with balance checking version)
- `PREFIX`: Target address prefix
- `SYNC_MODE`: Use synchronous or asynchronous balance checking
- `BATCH_SIZE`: CUDA batch size for address generation

## Tips

- Longer prefixes will take exponentially more time to find
- Consider using shorter prefixes for testing
- Multiple RPC endpoints provide redundancy and failover
- The system automatically tracks and saves best partial matches
- CUDA batch size is optimized for RTX series GPUs
- Balance checking automatically determines optimal batch sizes for each RPC

## Performance

The following metrics are based on a RTX 4060 Mobile GPU (8GB VRAM).

- The full cuda version without balance checking manages to check about 25,000,000 (25 million) addresses per second.
- The balance checking version manages to check about 2000 prefixes and 1000 balances per second. Disabling balance checking increases the speed to about 12,000 prefixes per second.
