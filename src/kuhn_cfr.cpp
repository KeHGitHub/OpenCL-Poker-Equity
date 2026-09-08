#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNumActions = 2;
constexpr int kPass = 0;
constexpr int kBet = 1;

struct State {
    std::array<int, 2> cards{};
    std::string history;

    bool terminal() const {
        return history == "pp"
            || history == "bp"
            || history == "bb"
            || history == "pbp"
            || history == "pbb";
    }

    int current_player() const {
        return static_cast<int>(history.size() % 2);
    }

    State next(int action) const {
        State child = *this;
        child.history += action == kPass ? 'p' : 'b';
        return child;
    }

    double utility_for_player0() const {
        const bool p0_wins_showdown = cards[0] > cards[1];

        if (history == "pp") {
            return p0_wins_showdown ? 1.0 : -1.0;
        }
        if (history == "bp") {
            return 1.0;
        }
        if (history == "pbp") {
            return -1.0;
        }
        if (history == "bb" || history == "pbb") {
            return p0_wins_showdown ? 2.0 : -2.0;
        }

        throw std::runtime_error("invalid terminal history: " + history);
    }
};

struct InfoSet {
    std::array<double, kNumActions> regret_sum{0.0, 0.0};
    std::array<double, kNumActions> strategy_sum{0.0, 0.0};

    std::array<double, kNumActions> strategy(double realization_weight) {
        std::array<double, kNumActions> result{};
        double total_positive_regret = 0.0;

        for (int action = 0; action < kNumActions; ++action) {
            result[action] = regret_sum[action] > 0.0 ? regret_sum[action] : 0.0;
            total_positive_regret += result[action];
        }

        if (total_positive_regret > 0.0) {
            for (int action = 0; action < kNumActions; ++action) {
                result[action] /= total_positive_regret;
            }
        } else {
            result = {0.5, 0.5};
        }

        for (int action = 0; action < kNumActions; ++action) {
            strategy_sum[action] += realization_weight * result[action];
        }

        return result;
    }

    std::array<double, kNumActions> average_strategy() const {
        const double total = strategy_sum[0] + strategy_sum[1];
        if (total <= 0.0) {
            return {0.5, 0.5};
        }
        return {strategy_sum[0] / total, strategy_sum[1] / total};
    }
};

std::map<std::string, InfoSet> infosets;

char card_name(int card) {
    switch (card) {
        case 0: return 'J';
        case 1: return 'Q';
        case 2: return 'K';
        default: return '?';
    }
}

std::string infoset_key(const State& state) {
    const int player = state.current_player();
    std::string key;
    key += card_name(state.cards[player]);
    key += state.history;
    return key;
}

std::string history_from_key(const std::string& key) {
    return key.size() <= 1 ? "" : key.substr(1);
}

const char* action_name(const std::string& history, int action) {
    const bool facing_bet = history == "b" || history == "pb";
    if (action == kPass) {
        return facing_bet ? "fold" : "check";
    }
    return facing_bet ? "call" : "bet";
}


// Because our value is always written from Player 0's perspective, and Player 1 wants Player 0's payoff low.
double cfr(State state, double p0, double p1) {
    if (state.terminal()) {
        return state.utility_for_player0();
    }

    // p0 and p1 are reach probabilities: how likely each player's own past
    // mixed-strategy choices made this state happen. When updating one
    // player's regret, CFR weights by the opponent's reach probability because
    // it asks: if I were at this infoset, how often did the opponent's choices
    // create this situation?
    const int player = state.current_player();
    InfoSet& info = infosets[infoset_key(state)];
    const auto strategy = info.strategy(player == 0 ? p0 : p1);

    std::array<double, kNumActions> action_value{};
    double node_value = 0.0;

    for (int action = 0; action < kNumActions; ++action) {
        if (player == 0) {
            action_value[action] = cfr(state.next(action), p0 * strategy[action], p1);
        } else {
            action_value[action] = cfr(state.next(action), p0, p1 * strategy[action]);
        }
        node_value += strategy[action] * action_value[action];
    }

    for (int action = 0; action < kNumActions; ++action) {
        if (player == 0) {
            info.regret_sum[action] += p1 * (action_value[action] - node_value);
        } else {
            info.regret_sum[action] += p0 * (node_value - action_value[action]);
        }
    }

    return node_value;
}

std::vector<std::array<int, 2>> all_deals() {
    return {
        {0, 1},
        {0, 2},
        {1, 0},
        {1, 2},
        {2, 0},
        {2, 1},
    };
}

double train(int iterations) {
    double total_utility = 0.0;
    const auto deals = all_deals();

    for (int i = 0; i < iterations; ++i) {
        double iteration_utility = 0.0;
        for (const auto& deal : deals) {
            iteration_utility += cfr(State{deal, ""}, 1.0, 1.0);
        }
        total_utility += iteration_utility / static_cast<double>(deals.size());
    }

    return total_utility / static_cast<double>(iterations);
}

void print_strategy() {
    std::cout << "\nAverage CFR strategy\n";
    std::cout << "--------------------\n";

    for (const auto& [key, info] : infosets) {
        const auto avg = info.average_strategy();
        const std::string history = history_from_key(key);

        std::cout << std::setw(4) << key << "  "
                  << std::setw(5) << action_name(history, kPass) << ": "
                  << std::setw(6) << std::fixed << std::setprecision(2) << avg[kPass] * 100.0 << "%  "
                  << std::setw(5) << action_name(history, kBet) << ": "
                  << std::setw(6) << avg[kBet] * 100.0 << "%\n";
    }
}

int sample_action(const std::array<double, kNumActions>& strategy, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng) < strategy[kPass] ? kPass : kBet;
}

int read_user_action(const std::string& history) {
    while (true) {
        std::cout << "Choose [" << action_name(history, kPass)
                  << "/" << action_name(history, kBet) << "] > ";

        std::string input;
        if (!std::getline(std::cin, input)) {
            return -1;
        }

        for (char& ch : input) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        if (input == "q" || input == "quit") {
            return -1;
        }
        if (input == "p" || input == "check" || input == "fold") {
            return kPass;
        }
        if (input == "b" || input == "bet" || input == "call") {
            return kBet;
        }

        std::cout << "Please type one action, or q to quit.\n";
    }
}

std::array<int, 2> random_deal(std::mt19937& rng) {
    std::array<int, 3> deck = {0, 1, 2};
    std::shuffle(deck.begin(), deck.end(), rng);
    return {deck[0], deck[1]};
}

void describe_terminal(const State& state) {
    const double utility = state.utility_for_player0();

    std::cout << "\nYour card:   " << card_name(state.cards[0]) << '\n';
    std::cout << "Solver card: " << card_name(state.cards[1]) << '\n';
    std::cout << "History:     " << state.history << '\n';

    if (utility > 0.0) {
        std::cout << "Result:      you win " << utility << "\n\n";
    } else {
        std::cout << "Result:      you lose " << -utility << "\n\n";
    }
}

void play_game(std::mt19937& rng) {
    std::cout << "\nPlay Kuhn poker against the solver.\n";
    std::cout << "Cards are J < Q < K. Each player antes 1. Type q to quit.\n\n";

    while (true) {
        State state{random_deal(rng), ""};
        std::cout << "New hand. Your card is " << card_name(state.cards[0]) << ".\n";

        while (!state.terminal()) {
            const int player = state.current_player();

            if (player == 0) {
                const int action = read_user_action(state.history);
                if (action < 0) {
                    return;
                }
                state = state.next(action);
            } else {
                const std::string key = infoset_key(state);
                const auto avg = infosets[key].average_strategy();
                const int action = sample_action(avg, rng);
                std::cout << "Solver chooses " << action_name(state.history, action) << ".\n";
                state = state.next(action);
            }
        }

        describe_terminal(state);
    }
}

int parse_iterations(int argc, char** argv) {
    int iterations = 100000;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iterations" || arg == "-n") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--iterations requires a value");
            }
            iterations = std::stoi(argv[++i]);
            if (iterations <= 0) {
                throw std::runtime_error("--iterations must be positive");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--iterations 100000]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return iterations;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int iterations = parse_iterations(argc, argv);
        const double ev = train(iterations);

        std::cout << "Trained Kuhn CFR for " << iterations << " iterations.\n";
        std::cout << "Estimated Player 0 EV: " << std::fixed << std::setprecision(6) << ev << '\n';
        std::cout << "Known equilibrium EV:  " << -1.0 / 18.0 << '\n';

        print_strategy();

        std::random_device rd;
        std::mt19937 rng(rd());
        play_game(rng);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
