#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

constexpr int NUM_ACTIONS = 2;

enum Action
{
    PASS = 0, // check or fold
    BET  = 1  // bet or call
};

char card_name(int card)
{
    switch (card)
    {
    case 0: return 'J';
    case 1: return 'Q';
    case 2: return 'K';
    default: return '?';
    }
}

// ============================================================
// Game state
// ============================================================

struct State
{
    std::array<int, 2> cards;
    std::string history;

    bool terminal() const
    {
        return history == "pp"  ||
               history == "bp"  ||
               history == "bb"  ||
               history == "pbp" ||
               history == "pbb";
    }

    int current_player() const
    {
        return static_cast<int>(history.size() % 2);
    }

    State next(int action) const
    {
        State child = *this;

        child.history +=
            (action == PASS) ? 'p' : 'b';

        return child;
    }

    bool player0_has_higher_card() const
    {
        return cards[0] > cards[1];
    }

    double utility_for_player0() const
    {
        // check-check
        if (history == "pp")
        {
            return player0_has_higher_card()
                ? 1.0
                : -1.0;
        }

        // P0 bet, P1 folded
        if (history == "bp")
        {
            return 1.0;
        }

        // P0 checked, P1 bet, P0 folded
        if (history == "pbp")
        {
            return -1.0;
        }

        // bet-call showdown
        if (history == "bb" ||
            history == "pbb")
        {
            return player0_has_higher_card()
                ? 2.0
                : -2.0;
        }

        std::cerr
            << "Invalid terminal history: "
            << history
            << '\n';

        return 0.0;
    }

    std::string infoset_key() const
    {
        const int player =
            current_player();

        // Player knows:
        //
        //   own card
        //   public betting history
        //
        // Player does NOT know opponent's card.

        std::string key;

        key += card_name(cards[player]);
        key += history;

        return key;
    }
};

// ============================================================
// Information set
// ============================================================

struct InfoSet
{
    // Used to determine the CURRENT strategy.
    std::array<double, NUM_ACTIONS> regret_sum{
        0.0, 0.0
    };

    // Used to determine the FINAL AVERAGE strategy.
    std::array<double, NUM_ACTIONS> strategy_sum{
        0.0, 0.0
    };

    // --------------------------------------------------------
    // CURRENT strategy
    //
    // Depends ONLY on regret_sum.
    // No reach probability involved here.
    // --------------------------------------------------------

    std::array<double, NUM_ACTIONS>
    current_strategy() const
    {
        std::array<double, NUM_ACTIONS> strategy{};

        double positive_regret_sum = 0.0;

        for (int a = 0; a < NUM_ACTIONS; ++a)
        {
            strategy[a] =
                regret_sum[a] > 0.0
                    ? regret_sum[a]
                    : 0.0;

            positive_regret_sum +=
                strategy[a];
        }

        if (positive_regret_sum > 0.0)
        {
            for (int a = 0; a < NUM_ACTIONS; ++a)
            {
                strategy[a] /=
                    positive_regret_sum;
            }
        }
        else
        {
            // No positive regret.
            // Use uniform strategy.
            for (int a = 0; a < NUM_ACTIONS; ++a)
            {
                strategy[a] =
                    1.0 / NUM_ACTIONS;
            }
        }

        return strategy;
    }

    // --------------------------------------------------------
    // Record the current strategy for the final average.
    //
    // own_reach_probability =
    //
    // probability that THIS player's own previous actions
    // caused them to reach this information set.
    // --------------------------------------------------------

    void accumulate_strategy(
        const std::array<double, NUM_ACTIONS>& strategy,
        double own_reach_probability)
    {
        for (int a = 0; a < NUM_ACTIONS; ++a)
        {
            strategy_sum[a] +=
                own_reach_probability
                * strategy[a];
        }
    }

    // --------------------------------------------------------
    // Final average CFR strategy
    // --------------------------------------------------------

    std::array<double, NUM_ACTIONS>
    average_strategy() const
    {
        std::array<double, NUM_ACTIONS> average{};

        double total =
            strategy_sum[PASS]
            +
            strategy_sum[BET];

        if (total > 0.0)
        {
            average[PASS] =
                strategy_sum[PASS] / total;

            average[BET] =
                strategy_sum[BET] / total;
        }
        else
        {
            average[PASS] = 0.5;
            average[BET]  = 0.5;
        }

        return average;
    }
};

// ============================================================
// All information sets
// ============================================================

std::map<std::string, InfoSet> infosets;

InfoSet& get_infoset(const State& state)
{
    return infosets[state.infoset_key()];
}

// ============================================================
// CFR
// ============================================================

double cfr(
    const State& state,
    double p0,
    double p1)
{
    // --------------------------------------------------------
    // 1. Terminal state
    // --------------------------------------------------------

    if (state.terminal())
    {
        return state.utility_for_player0();
    }

    // --------------------------------------------------------
    // 2. Current player
    // --------------------------------------------------------

    const int player =
        state.current_player();

    InfoSet& info =
        get_infoset(state);

    // --------------------------------------------------------
    // 3. Calculate CURRENT strategy
    //
    // Important:
    //
    // This depends ONLY on regret_sum.
    // --------------------------------------------------------

    const auto strategy =
        info.current_strategy();

    // --------------------------------------------------------
    // 4. Record current strategy for final average
    //
    // Use this player's OWN reach probability.
    // --------------------------------------------------------

    const double own_reach_probability =
        (player == 0)
            ? p0
            : p1;

    info.accumulate_strategy(
        strategy,
        own_reach_probability
    );

    // Everything in this implementation is stored
    // from Player 0's utility perspective.

    std::array<double, NUM_ACTIONS>
        action_value{};

    double node_value = 0.0;

    // --------------------------------------------------------
    // 5. Evaluate BOTH actions
    // --------------------------------------------------------

    for (int a = 0; a < NUM_ACTIONS; ++a)
    {
        State child =
            state.next(a);

        if (player == 0)
        {
            // Player 0 chose action a,
            // so only P0's reach probability changes.

            action_value[a] =
                cfr(
                    child,
                    p0 * strategy[a],
                    p1
                );
        }
        else
        {
            // Player 1 chose action a,
            // so only P1's reach probability changes.

            action_value[a] =
                cfr(
                    child,
                    p0,
                    p1 * strategy[a]
                );
        }

        // Expected value of current mixed strategy.
        node_value +=
            strategy[a]
            * action_value[a];
    }

    // --------------------------------------------------------
    // 6. Update counterfactual regret
    // --------------------------------------------------------

    for (int a = 0; a < NUM_ACTIONS; ++a)
    {
        if (player == 0)
        {
            // Player 0 wants higher P0 utility.
            //
            // "How much better would action a have been
            // compared with my current mixed strategy?"

            const double regret =
                action_value[a]
                -
                node_value;

            // COUNTERFACTUAL weighting:
            //
            // Use OPPONENT reach probability.
            info.regret_sum[a] +=
                p1 * regret;
        }
        else
        {
            // Player 1 wants LOWER Player-0 utility.
            //
            // Therefore reverse the comparison.

            const double regret =
                node_value
                -
                action_value[a];

            // Again use OPPONENT reach probability.
            //
            // For Player 1, opponent = Player 0.
            info.regret_sum[a] +=
                p0 * regret;
        }
    }

    return node_value;
}

// ============================================================
// Printing
// ============================================================

std::string get_history_from_key(
    const std::string& key)
{
    if (key.size() <= 1)
    {
        return "";
    }

    return key.substr(1);
}

const char* action0_name(
    const std::string& history)
{
    // Facing a bet:
    //
    // b  = P0 bet, P1 deciding
    // pb = P0 check, P1 bet, P0 deciding

    if (history == "b" ||
        history == "pb")
    {
        return "FOLD";
    }

    return "CHECK";
}

const char* action1_name(
    const std::string& history)
{
    if (history == "b" ||
        history == "pb")
    {
        return "CALL";
    }

    return "BET";
}

void print_strategy()
{
    std::cout
        << "\n====================================\n"
        << "AVERAGE CFR STRATEGY\n"
        << "====================================\n\n";

    for (const auto& [key, info] : infosets)
    {
        const auto strategy =
            info.average_strategy();

        const std::string history =
            get_history_from_key(key);

        std::cout
            << "Information set: "
            << key
            << '\n';

        std::cout
            << "  "
            << action0_name(history)
            << ": "
            << std::fixed
            << std::setprecision(2)
            << strategy[PASS] * 100.0
            << "%\n";

        std::cout
            << "  "
            << action1_name(history)
            << ": "
            << strategy[BET] * 100.0
            << "%\n\n";
    }
}

// ============================================================
// Main
// ============================================================

int main()
{
    // All six possible ordered deals.
    //
    // cards[0] = Player 0
    // cards[1] = Player 1

    const std::vector<std::array<int, 2>> deals{
        {0, 1}, // J Q
        {0, 2}, // J K

        {1, 0}, // Q J
        {1, 2}, // Q K

        {2, 0}, // K J
        {2, 1}  // K Q
    };

    constexpr int ITERATIONS =
        100000;

    double total_utility = 0.0;

    for (int iteration = 0;
         iteration < ITERATIONS;
         ++iteration)
    {
        double iteration_utility = 0.0;

        // Chance node:
        //
        // Every deal is equally likely.

        for (const auto& deal : deals)
        {
            State initial_state{
                deal,
                ""
            };

            // At the root:
            //
            // P0 has taken no actions yet:
            // p0 = 1
            //
            // P1 has taken no actions yet:
            // p1 = 1

            iteration_utility +=
                cfr(
                    initial_state,
                    1.0,
                    1.0
                );
        }

        iteration_utility /=
            static_cast<double>(
                deals.size()
            );

        total_utility +=
            iteration_utility;
    }

    const double estimated_game_value =
        total_utility
        /
        static_cast<double>(
            ITERATIONS
        );

    std::cout
        << "Iterations: "
        << ITERATIONS
        << '\n';

    std::cout
        << "Estimated Player 0 EV: "
        << std::fixed
        << std::setprecision(6)
        << estimated_game_value
        << '\n';

    std::cout
        << "Known Kuhn Poker equilibrium EV: "
        << -1.0 / 18.0
        << '\n';

    print_strategy();

    return 0;
}