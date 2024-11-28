CUDA_CRYPTO_CODE = """
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define KECCAK_ROUNDS 24
#define BATCH_SIZE 65536

// Keccak round constants
__device__ __constant__ uint64_t keccak_round_constants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Keccak state rotation offsets
__device__ __constant__ int keccak_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Keccak state permutation indices
__device__ __constant__ int keccak_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// Keccak-256 hash function
__device__ void keccak256_transform(uint64_t* state) {
    uint64_t temp, C[5];
    int i, j;

    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta step
        for (i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }
        for (i = 0; i < 5; i++) {
            temp = C[(i + 4) % 5] ^ ((C[(i + 1) % 5] << 1) | (C[(i + 1) % 5] >> 63));
            for (j = 0; j < 25; j += 5) {
                state[j + i] ^= temp;
            }
        }

        // Rho and Pi steps
        temp = state[1];
        for (i = 0; i < 24; i++) {
            j = keccak_piln[i];
            C[0] = state[j];
            state[j] = ((temp << keccak_rotc[i]) | (temp >> (64 - keccak_rotc[i])));
            temp = C[0];
        }

        // Chi step
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++) {
                C[i] = state[j + i];
            }
            for (i = 0; i < 5; i++) {
                state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
            }
        }

        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

__device__ void keccak256_update(uint64_t* state, const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        state[i/8] ^= ((uint64_t)data[i]) << (8 * (i % 8));
    }
    keccak256_transform(state);
}

__device__ void keccak256_final(uint64_t* state, uint8_t* hash) {
    keccak256_transform(state);
    for (int i = 0; i < 4; i++) {
        ((uint64_t*)hash)[i] = state[i];
    }
}

// Main kernel for address generation and checking
extern "C" __global__ void generate_and_check(
    uint8_t* private_keys,
    uint8_t* addresses,
    int* match_lengths,
    const uint8_t* target_prefix,
    int prefix_len,
    uint32_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE) return;

    // Initialize random state
    curandState rng_state;
    curand_init(seed + idx, 0, 0, &rng_state);

    // Generate random private key
    uint8_t private_key[32];
    for (int i = 0; i < 32; i++) {
        private_key[i] = curand(&rng_state) & 0xFF;
    }

    // Store private key
    for (int i = 0; i < 32; i++) {
        private_keys[idx * 32 + i] = private_key[i];
    }

    // Initialize Keccak state
    uint64_t keccak_state[25] = {0};
    
    // Hash private key to get address
    keccak256_update(keccak_state, private_key, 32);
    uint8_t hash[32];
    keccak256_final(keccak_state, hash);

    // Take last 20 bytes as address
    for (int i = 0; i < 20; i++) {
        addresses[idx * 20 + i] = hash[i + 12];
    }

    // Convert address to hex and compare with target
    int match_count = 0;
    for (int i = 0; i < prefix_len && i < 20; i++) {
        uint8_t addr_byte = addresses[idx * 20 + i];
        uint8_t target_byte = target_prefix[i];
        
        // Convert each byte to two hex characters
        char addr_hex[2];
        addr_hex[0] = (addr_byte >> 4) <= 9 ? (addr_byte >> 4) + '0' : (addr_byte >> 4) - 10 + 'a';
        addr_hex[1] = (addr_byte & 0x0F) <= 9 ? (addr_byte & 0x0F) + '0' : (addr_byte & 0x0F) - 10 + 'a';
        
        char target_hex[2];
        target_hex[0] = (target_byte >> 4) <= 9 ? (target_byte >> 4) + '0' : (target_byte >> 4) - 10 + 'a';
        target_hex[1] = (target_byte & 0x0F) <= 9 ? (target_byte & 0x0F) + '0' : (target_byte & 0x0F) - 10 + 'a';
        
        // Compare characters
        if (addr_hex[0] != target_hex[0]) {
            break;  // First character doesn't match
        }
        match_count++;
        
        if (addr_hex[1] != target_hex[1]) {
            break;  // Second character doesn't match
        }
        match_count++;
    }

    match_lengths[idx] = match_count;
}
"""
