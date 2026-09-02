enum {
    MAX_OPPONENTS = 22
};

uint next_random(uint* state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

uchar draw_card(ulong* used, uint* state) {
    while (1) {
        uint card = next_random(state) % 52u;
        ulong bit = ((ulong)1) << card;
        if (((*used) & bit) == 0ul) {
            *used |= bit;
            return (uchar)card;
        }
    }
}

void mark_card(ulong* used, uchar card) {
    *used |= ((ulong)1) << ((uint)card);
}

uint pack_score(int category, int a, int b, int c, int d, int e) {
    return (((uint)category) << 20)
        | (((uint)a) << 16)
        | (((uint)b) << 12)
        | (((uint)c) << 8)
        | (((uint)d) << 4)
        | ((uint)e);
}

int straight_high(int rank_mask) {
    for (int high = 12; high >= 4; --high) {
        int needed = 0x1f << (high - 4);
        if ((rank_mask & needed) == needed) {
            return high;
        }
    }

    if ((rank_mask & ((1 << 12) | 0x0f)) == ((1 << 12) | 0x0f)) {
        return 3;
    }

    return -1;
}

uint eval5(uchar c0, uchar c1, uchar c2, uchar c3, uchar c4) {
    uchar cards[5] = {c0, c1, c2, c3, c4};
    int rank_counts[13];
    int suit_counts[4];

    for (int i = 0; i < 13; ++i) {
        rank_counts[i] = 0;
    }
    for (int i = 0; i < 4; ++i) {
        suit_counts[i] = 0;
    }

    int rank_mask = 0;
    for (int i = 0; i < 5; ++i) {
        int rank = ((int)cards[i]) / 4;
        int suit = ((int)cards[i]) % 4;
        ++rank_counts[rank];
        ++suit_counts[suit];
        rank_mask |= 1 << rank;
    }

    int is_flush = 0;
    for (int suit = 0; suit < 4; ++suit) {
        if (suit_counts[suit] == 5) {
            is_flush = 1;
        }
    }

    int straight = straight_high(rank_mask);
    if (is_flush && straight >= 0) {
        return pack_score(8, straight, 0, 0, 0, 0);
    }

    int quad = -1;
    int trips = -1;
    int pairs[2] = {-1, -1};
    int pair_count = 0;

    for (int rank = 12; rank >= 0; --rank) {
        if (rank_counts[rank] == 4) {
            quad = rank;
        } else if (rank_counts[rank] == 3) {
            trips = rank;
        } else if (rank_counts[rank] == 2) {
            pairs[pair_count] = rank;
            ++pair_count;
        }
    }

    if (quad >= 0) {
        int kicker = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kicker = rank;
                break;
            }
        }
        return pack_score(7, quad, kicker, 0, 0, 0);
    }

    if (trips >= 0 && pair_count > 0) {
        return pack_score(6, trips, pairs[0], 0, 0, 0);
    }

    if (is_flush) {
        int ranks[5];
        int out = 0;
        for (int rank = 12; rank >= 0; --rank) {
            for (int i = 0; i < rank_counts[rank]; ++i) {
                ranks[out] = rank;
                ++out;
            }
        }
        return pack_score(5, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]);
    }

    if (straight >= 0) {
        return pack_score(4, straight, 0, 0, 0, 0);
    }

    if (trips >= 0) {
        int kickers[2];
        int out = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kickers[out] = rank;
                ++out;
            }
        }
        return pack_score(3, trips, kickers[0], kickers[1], 0, 0);
    }

    if (pair_count == 2) {
        int kicker = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kicker = rank;
                break;
            }
        }
        return pack_score(2, pairs[0], pairs[1], kicker, 0, 0);
    }

    if (pair_count == 1) {
        int kickers[3];
        int out = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kickers[out] = rank;
                ++out;
            }
        }
        return pack_score(1, pairs[0], kickers[0], kickers[1], kickers[2], 0);
    }

    int ranks[5];
    int out = 0;
    for (int rank = 12; rank >= 0; --rank) {
        if (rank_counts[rank] == 1) {
            ranks[out] = rank;
            ++out;
        }
    }
    return pack_score(0, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]);
}

uint eval7(uchar cards[7]) {
    uint best = 0;

    for (int a = 0; a < 3; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            for (int c = b + 1; c < 5; ++c) {
                for (int d = c + 1; d < 6; ++d) {
                    for (int e = d + 1; e < 7; ++e) {
                        uint score = eval5(cards[a], cards[b], cards[c], cards[d], cards[e]);
                        if (score > best) {
                            best = score;
                        }
                    }
                }
            }
        }
    }

    return best;
}

__kernel void simulate_equity(
    __global const uchar* hero,
    __global const uchar* board,
    uint board_count,
    uint opponent_count,
    ulong seed,
    __global uchar* outcomes,
    __global uchar* pot_shares) {
    size_t gid = get_global_id(0);
    uint state = (uint)(seed ^ (seed >> 32) ^ (((ulong)gid + 1ul) * 747796405ul));
    if (state == 0u) {
        state = 2891336453u;
    }

    ulong used = 0ul;
    mark_card(&used, hero[0]);
    mark_card(&used, hero[1]);

    uchar final_board[5];
    for (uint i = 0; i < board_count; ++i) {
        final_board[i] = board[i];
        mark_card(&used, board[i]);
    }

    uchar opponents[MAX_OPPONENTS * 2];
    for (uint i = 0; i < opponent_count; ++i) {
        opponents[i * 2] = draw_card(&used, &state);
        opponents[i * 2 + 1] = draw_card(&used, &state);
    }

    for (uint i = board_count; i < 5; ++i) {
        final_board[i] = draw_card(&used, &state);
    }

    uchar hero_cards[7] = {
        hero[0],
        hero[1],
        final_board[0],
        final_board[1],
        final_board[2],
        final_board[3],
        final_board[4]
    };

    uint hero_score = eval7(hero_cards);
    uint split_count = 1;

    for (uint i = 0; i < opponent_count; ++i) {
        uchar villain_cards[7] = {
            opponents[i * 2],
            opponents[i * 2 + 1],
            final_board[0],
            final_board[1],
            final_board[2],
            final_board[3],
            final_board[4]
        };

        uint villain_score = eval7(villain_cards);
        if (villain_score > hero_score) {
            outcomes[gid] = 0;
            pot_shares[gid] = 0;
            return;
        }
        if (villain_score == hero_score) {
            ++split_count;
        }
    }

    if (split_count == 1) {
        outcomes[gid] = 2;
        pot_shares[gid] = 1;
    } else {
        outcomes[gid] = 1;
        pot_shares[gid] = (uchar)split_count;
    }
}
