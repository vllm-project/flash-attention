#pragma once

#include <stdint.h>

#include "cute/container/alignment.hpp"

struct CUTE_ALIGNAS(16) StreamKWorkTile {
    int const m_block = -1;
    int const n_block_start = 0;
    uint16_t const n_blocks = 0; // Max n_blocks per tile is 65535
    uint16_t const bidb = 0;     // Max batch size is 65535
    uint16_t const bidh = 0;     // Max num heads is 65535
    uint8_t const peer_id = 0;   // Max 255 peers
    uint8_t const num_peers = 0; // Max 255 peers
};

struct CUTE_ALIGNAS(8) StreamKCombineTile {
    int const m_block;
    uint16_t const bidh;
    uint16_t const bidb;
    uint16_t const num_peers; // Number of peers / num_splits
};
