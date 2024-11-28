# Ethereum Vanity Address Generator with CUDA

A high-performance Ethereum vanity address generator that uses CUDA GPU acceleration to quickly generate Ethereum addresses matching a desired prefix pattern. It also includes optional balance checking functionality across multiple RPC endpoints.

## Features

- 🚀 CUDA GPU acceleration for fast address generation
- 🔍 Configurable prefix matching with support for hex characters
- 💰 Optional balance checking across multiple RPC endpoints
- ⚡ Batch processing for efficient balance checks
- 🔄 Automatic RPC failover and rate limit handling
- 📊 Real-time status updates and progress tracking
- 💾 Automatic saving of addresses with balances to file

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
    # RPC URL for Ethereum node
    RPC_URL=https://eth.llamarpc.com
    # How many addresses to check in each batch
    BALANCE_BATCH_SIZE=100
    # Desired address prefix
    PREFIX=dEAD000000000000000042069420694206942069
    SYNC_MODE=false  # Set to true for synchronous balance checking
    ```

## Usage

1. Run the script:
    ```bash
    python main.py
    ```

2. Enter your desired address prefix when prompted, or configure it in the `.env` file.

The script will begin generating addresses and checking balances if enabled. Status updates are printed every 10 seconds showing:

- Time elapsed
- Total attempts
- Generation speed
- Balance check status
- Best match found so far

When a matching address is found, it will be displayed along with its private key.

## Configuration

The following settings can be configured in the `.env` file:

- `CHECK_BALANCES`: Enable/disable balance checking
- `RPC_URL`: Ethereum RPC endpoint URL
- `BALANCE_BATCH_SIZE`: Number of addresses to check in each batch
- `PREFIX`: Target address prefix
- `SYNC_MODE`: Use synchronous or asynchronous balance checking

## Tips

- Longer prefixes will take exponentially more time to find
- Consider using shorter prefixes for testing
- Multiple RPC endpoints are used for redundancy
- Found addresses with balances are saved to `found.txt`
