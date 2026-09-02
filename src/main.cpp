#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace {

constexpr std::uint32_t kMaxOpponents = 22;

struct Options {
    std::vector<unsigned char> hand;
    std::vector<unsigned char> board;
    std::uint32_t opponents = 1;
    std::uint64_t trials = 1'000'000;
    std::uint64_t seed = 0;
    bool list_devices = false;
    bool profile = false;
    std::uint32_t profile_runs = 5;
    std::uint32_t warmup_runs = 1;
    bool hand_provided = false;
    bool board_provided = false;
};

struct OpenCLDevice {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};

struct OpenCLRuntime {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::string device_name;
};

struct SimulationResult {
    std::uint64_t wins = 0;
    std::uint64_t ties = 0;
    std::uint64_t losses = 0;
    double pot_share_total = 0.0;
};

struct TimingSummary {
    double min_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
};

SimulationResult run_opencl_simulation(OpenCLRuntime& runtime, const Options& options);

std::string cl_error_name(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        default: return "OpenCL error " + std::to_string(err);
    }
}

void check(cl_int err, const std::string& what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(what + ": " + cl_error_name(err));
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open " + path);
    }

    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

std::string device_info_string(cl_device_id device, cl_device_info key) {
    size_t size = 0;
    check(clGetDeviceInfo(device, key, 0, nullptr, &size), "clGetDeviceInfo size");

    std::string value(size, '\0');
    check(clGetDeviceInfo(device, key, size, value.data(), nullptr), "clGetDeviceInfo value");

    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::string platform_info_string(cl_platform_id platform, cl_platform_info key) {
    size_t size = 0;
    check(clGetPlatformInfo(platform, key, 0, nullptr, &size), "clGetPlatformInfo size");

    std::string value(size, '\0');
    check(clGetPlatformInfo(platform, key, size, value.data(), nullptr), "clGetPlatformInfo value");

    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::vector<cl_device_id> devices_for(cl_platform_id platform, cl_device_type type) {
    cl_device_id device = nullptr;
    cl_int err = clGetDeviceIDs(platform, type, 1, &device, nullptr);
    if (err == CL_DEVICE_NOT_FOUND || err == CL_INVALID_VALUE) {
        return {};
    }
    check(err, "clGetDeviceIDs list");

    return {device};
}

std::vector<cl_device_id> devices_from_context_type(cl_device_type type) {
    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContextFromType(nullptr, type, nullptr, nullptr, &err);
    if (err == CL_DEVICE_NOT_FOUND || err == CL_INVALID_VALUE) {
        return {};
    }
    check(err, "clCreateContextFromType");

    size_t bytes = 0;
    check(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &bytes), "clGetContextInfo devices size");
    std::vector<cl_device_id> devices(bytes / sizeof(cl_device_id));
    check(
        clGetContextInfo(context, CL_CONTEXT_DEVICES, bytes, devices.data(), nullptr),
        "clGetContextInfo devices");
    clReleaseContext(context);

    return devices;
}

OpenCLDevice choose_device() {
    const std::array<cl_device_type, 2> device_order = {
        static_cast<cl_device_type>(CL_DEVICE_TYPE_GPU),
        static_cast<cl_device_type>(CL_DEVICE_TYPE_DEFAULT),
    };

    for (const cl_device_type type : device_order) {
        const auto context_devices = devices_from_context_type(type);
        if (!context_devices.empty()) {
            return {nullptr, context_devices.front()};
        }

        const auto default_platform_devices = devices_for(nullptr, type);
        if (!default_platform_devices.empty()) {
            return {nullptr, default_platform_devices.front()};
        }
    }

    cl_uint platform_count = 0;
    check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
    if (platform_count == 0) {
        throw std::runtime_error("no OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs list");

    for (const cl_device_type type : device_order) {
        for (cl_platform_id platform : platforms) {
            const auto devices = devices_for(platform, type);
            if (!devices.empty()) {
                return {platform, devices.front()};
            }
        }
    }

    throw std::runtime_error("no OpenCL GPU/default device found");
}

OpenCLRuntime create_opencl_runtime() {
    const OpenCLDevice selected = choose_device();
    OpenCLRuntime runtime;
    runtime.device_name = device_info_string(selected.device, CL_DEVICE_NAME);

    cl_int err = CL_SUCCESS;
    runtime.context = clCreateContext(nullptr, 1, &selected.device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

    runtime.queue = clCreateCommandQueue(runtime.context, selected.device, 0, &err);
    check(err, "clCreateCommandQueue");

    const std::string source = read_text_file("kernels/equity_kernel.cl");
    const char* source_ptr = source.c_str();
    const size_t source_size = source.size();
    runtime.program = clCreateProgramWithSource(runtime.context, 1, &source_ptr, &source_size, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(runtime.program, 1, &selected.device, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(runtime.program, selected.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(runtime.program, selected.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        throw std::runtime_error("OpenCL program build failed:\n" + log);
    }

    runtime.kernel = clCreateKernel(runtime.program, "simulate_equity", &err);
    check(err, "clCreateKernel simulate_equity");

    return runtime;
}

void release_opencl_runtime(OpenCLRuntime& runtime) {
    if (runtime.kernel != nullptr) {
        clReleaseKernel(runtime.kernel);
        runtime.kernel = nullptr;
    }
    if (runtime.program != nullptr) {
        clReleaseProgram(runtime.program);
        runtime.program = nullptr;
    }
    if (runtime.queue != nullptr) {
        clReleaseCommandQueue(runtime.queue);
        runtime.queue = nullptr;
    }
    if (runtime.context != nullptr) {
        clReleaseContext(runtime.context);
        runtime.context = nullptr;
    }
}

int parse_rank(char value) {
    const std::string ranks = "23456789TJQKA";
    const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(value)));
    const auto pos = ranks.find(upper);
    if (pos == std::string::npos) {
        throw std::runtime_error(std::string("invalid card rank: ") + value);
    }
    return static_cast<int>(pos);
}

int parse_suit(char value) {
    switch (std::tolower(static_cast<unsigned char>(value))) {
        case 'c': return 0;
        case 'd': return 1;
        case 'h': return 2;
        case 's': return 3;
        default:
            throw std::runtime_error(std::string("invalid card suit: ") + value);
    }
}

std::string card_to_string(unsigned char card) {
    const std::string ranks = "23456789TJQKA";
    const std::string suits = "cdhs";
    std::string out;
    out.push_back(ranks[card / 4]);
    out.push_back(suits[card % 4]);
    return out;
}

std::vector<std::string> card_tokens(std::string input) {
    for (char& ch : input) {
        if (ch == ',' || ch == ';' || ch == '/') {
            ch = ' ';
        }
    }

    std::vector<std::string> tokens;
    std::istringstream stream(input);
    for (std::string token; stream >> token;) {
        if (token.size() > 2 && token.size() % 2 == 0) {
            for (size_t i = 0; i < token.size(); i += 2) {
                tokens.push_back(token.substr(i, 2));
            }
        } else {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<unsigned char> parse_cards(const std::string& input) {
    std::vector<unsigned char> cards;
    for (const std::string& token : card_tokens(input)) {
        if (token.size() != 2) {
            throw std::runtime_error("invalid card token: " + token);
        }

        const int rank = parse_rank(token[0]);
        const int suit = parse_suit(token[1]);
        cards.push_back(static_cast<unsigned char>(rank * 4 + suit));
    }
    return cards;
}

std::uint64_t parse_u64(const std::string& text, const std::string& name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        throw std::runtime_error("invalid " + name + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --hand \"As Ks\" [--board \"Qh Jh 2c\"] [--opponents 1] [--trials 1000000] [--seed 123]\n\n"
        << "  " << argv0 << " --profile [--hand \"As Ks\"] [--board \"Qh Jh 2c\"] [--profile-runs 5] [--warmup-runs 1]\n\n"
        << "  " << argv0 << " --list-devices\n\n"
        << "Cards use rank+suit notation: ranks 2-9,T,J,Q,K,A and suits c,d,h,s.\n"
        << "Each opponent hand is sampled from the unknown cards.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--hand" || arg == "-h") {
            options.hand = parse_cards(require_value(arg));
            options.hand_provided = true;
        } else if (arg == "--board" || arg == "-b") {
            options.board = parse_cards(require_value(arg));
            options.board_provided = true;
        } else if (arg == "--opponents" || arg == "-o") {
            const auto opponents = parse_u64(require_value(arg), "opponent count");
            if (opponents > kMaxOpponents) {
                throw std::runtime_error("--opponents cannot exceed " + std::to_string(kMaxOpponents));
            }
            options.opponents = static_cast<std::uint32_t>(opponents);
        } else if (arg == "--trials" || arg == "-n") {
            options.trials = parse_u64(require_value(arg), "trial count");
        } else if (arg == "--seed") {
            options.seed = parse_u64(require_value(arg), "seed");
        } else if (arg == "--profile") {
            options.profile = true;
        } else if (arg == "--profile-runs") {
            const auto runs = parse_u64(require_value(arg), "profile run count");
            if (runs == 0) {
                throw std::runtime_error("--profile-runs must be greater than zero");
            }
            options.profile_runs = static_cast<std::uint32_t>(runs);
        } else if (arg == "--warmup-runs") {
            options.warmup_runs = static_cast<std::uint32_t>(
                parse_u64(require_value(arg), "warmup run count"));
        } else if (arg == "--list-devices") {
            options.list_devices = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.list_devices) {
        return options;
    }

    if (!options.profile && options.hand.size() != 2) {
        throw std::runtime_error("--hand must contain exactly two cards");
    }
    if (options.hand_provided && options.hand.size() != 2) {
        throw std::runtime_error("--hand must contain exactly two cards");
    }
    if (options.board.size() > 5) {
        throw std::runtime_error("--board cannot contain more than five cards");
    }
    if (options.opponents == 0) {
        throw std::runtime_error("--opponents must be greater than zero");
    }
    if (options.trials == 0) {
        throw std::runtime_error("--trials must be greater than zero");
    }

    std::array<bool, 52> seen{};
    for (unsigned char card : options.hand) {
        if (seen[card]) {
            throw std::runtime_error("duplicate card: " + card_to_string(card));
        }
        seen[card] = true;
    }
    for (unsigned char card : options.board) {
        if (seen[card]) {
            throw std::runtime_error("duplicate card: " + card_to_string(card));
        }
        seen[card] = true;
    }

    if (options.seed == 0) {
        options.seed = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    return options;
}

std::string cards_to_string(const std::vector<unsigned char>& cards) {
    if (cards.empty()) {
        return "(none)";
    }

    std::ostringstream out;
    for (size_t i = 0; i < cards.size(); ++i) {
        if (i != 0) {
            out << ' ';
        }
        out << card_to_string(cards[i]);
    }
    return out.str();
}

cl_mem make_buffer(
    cl_context context,
    cl_mem_flags flags,
    size_t bytes,
    void* host_ptr,
    const std::string& name) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context, flags, bytes, host_ptr, &err);
    check(err, "clCreateBuffer " + name);
    return buffer;
}

void print_results(
    const Options& options,
    const SimulationResult& result,
    const std::string& device_name) {
    const double win_probability = static_cast<double>(result.wins) / static_cast<double>(options.trials);
    const double tie_probability = static_cast<double>(result.ties) / static_cast<double>(options.trials);
    const double loss_probability = static_cast<double>(result.losses) / static_cast<double>(options.trials);
    const double equity = result.pot_share_total / static_cast<double>(options.trials);

    std::cout << "Hero hand: " << cards_to_string(options.hand) << '\n';
    std::cout << "Board:     " << cards_to_string(options.board) << '\n';
    std::cout << "Opponents: " << options.opponents << " unknown random hand";
    if (options.opponents != 1) {
        std::cout << 's';
    }
    std::cout << '\n';
    std::cout << "Trials:    " << options.trials << '\n';
    std::cout << "Seed:      " << options.seed << '\n';
    std::cout << "Device:    " << device_name << "\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Win:       " << (win_probability * 100.0) << "%\n";
    std::cout << "Tie:       " << (tie_probability * 100.0) << "%\n";
    std::cout << "Loss:      " << (loss_probability * 100.0) << "%\n";
    std::cout << "Equity:    " << (equity * 100.0) << "%\n";
}

SimulationResult reduce_outcomes(
    const Options& options,
    const std::vector<unsigned char>& outcomes,
    const std::vector<unsigned char>& pot_shares) {
    SimulationResult result;
    for (unsigned char outcome : outcomes) {
        if (outcome == 2) {
            ++result.wins;
        } else if (outcome == 1) {
            ++result.ties;
        }
    }
    for (unsigned char share : pot_shares) {
        if (share > 0) {
            result.pot_share_total += 1.0 / static_cast<double>(share);
        }
    }
    result.losses = options.trials - result.wins - result.ties;
    return result;
}

int straight_high_reference(int rank_mask) {
    for (int high = 12; high >= 4; --high) {
        const int needed = 0x1f << (high - 4);
        if ((rank_mask & needed) == needed) {
            return high;
        }
    }

    if ((rank_mask & ((1 << 12) | 0x0f)) == ((1 << 12) | 0x0f)) {
        return 3;
    }

    return -1;
}

std::uint32_t pack_score_reference(int category, int a, int b, int c, int d, int e) {
    return (static_cast<std::uint32_t>(category) << 20)
        | (static_cast<std::uint32_t>(a) << 16)
        | (static_cast<std::uint32_t>(b) << 12)
        | (static_cast<std::uint32_t>(c) << 8)
        | (static_cast<std::uint32_t>(d) << 4)
        | static_cast<std::uint32_t>(e);
}

std::uint32_t eval5_reference(
    unsigned char c0,
    unsigned char c1,
    unsigned char c2,
    unsigned char c3,
    unsigned char c4) {
    const std::array<unsigned char, 5> cards = {c0, c1, c2, c3, c4};
    std::array<int, 13> rank_counts{};
    std::array<int, 4> suit_counts{};
    int rank_mask = 0;

    for (unsigned char card : cards) {
        const int rank = card / 4;
        const int suit = card % 4;
        ++rank_counts[rank];
        ++suit_counts[suit];
        rank_mask |= 1 << rank;
    }

    const bool is_flush = std::any_of(suit_counts.begin(), suit_counts.end(), [](int count) {
        return count == 5;
    });
    const int straight = straight_high_reference(rank_mask);
    if (is_flush && straight >= 0) {
        return pack_score_reference(8, straight, 0, 0, 0, 0);
    }

    int quad = -1;
    int trips = -1;
    std::array<int, 2> pairs = {-1, -1};
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
        return pack_score_reference(7, quad, kicker, 0, 0, 0);
    }

    if (trips >= 0 && pair_count > 0) {
        return pack_score_reference(6, trips, pairs[0], 0, 0, 0);
    }

    if (is_flush) {
        std::array<int, 5> ranks{};
        int out = 0;
        for (int rank = 12; rank >= 0; --rank) {
            for (int i = 0; i < rank_counts[rank]; ++i) {
                ranks[out++] = rank;
            }
        }
        return pack_score_reference(5, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]);
    }

    if (straight >= 0) {
        return pack_score_reference(4, straight, 0, 0, 0, 0);
    }

    if (trips >= 0) {
        std::array<int, 2> kickers{};
        int out = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kickers[out++] = rank;
            }
        }
        return pack_score_reference(3, trips, kickers[0], kickers[1], 0, 0);
    }

    if (pair_count == 2) {
        int kicker = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kicker = rank;
                break;
            }
        }
        return pack_score_reference(2, pairs[0], pairs[1], kicker, 0, 0);
    }

    if (pair_count == 1) {
        std::array<int, 3> kickers{};
        int out = 0;
        for (int rank = 12; rank >= 0; --rank) {
            if (rank_counts[rank] == 1) {
                kickers[out++] = rank;
            }
        }
        return pack_score_reference(1, pairs[0], kickers[0], kickers[1], kickers[2], 0);
    }

    std::array<int, 5> ranks{};
    int out = 0;
    for (int rank = 12; rank >= 0; --rank) {
        if (rank_counts[rank] == 1) {
            ranks[out++] = rank;
        }
    }
    return pack_score_reference(0, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]);
}

std::uint32_t eval7_reference(const std::array<unsigned char, 7>& cards) {
    std::uint32_t best = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            for (int c = b + 1; c < 5; ++c) {
                for (int d = c + 1; d < 6; ++d) {
                    for (int e = d + 1; e < 7; ++e) {
                        const std::uint32_t score = eval5_reference(
                            cards[a],
                            cards[b],
                            cards[c],
                            cards[d],
                            cards[e]);
                        best = std::max(best, score);
                    }
                }
            }
        }
    }
    return best;
}

std::uint32_t next_random_reference(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

unsigned char draw_card_reference(std::uint64_t& used, std::uint32_t& state) {
    while (true) {
        const auto card = static_cast<unsigned char>(next_random_reference(state) % 52);
        const std::uint64_t bit = std::uint64_t{1} << card;
        if ((used & bit) == 0) {
            used |= bit;
            return card;
        }
    }
}

void mark_card_reference(std::uint64_t& used, unsigned char card) {
    used |= std::uint64_t{1} << card;
}

SimulationResult run_cpu_reference(const Options& options) {
    SimulationResult result;

    for (std::uint64_t trial = 0; trial < options.trials; ++trial) {
        std::uint32_t state = static_cast<std::uint32_t>(
            options.seed ^ (options.seed >> 32) ^ ((trial + 1) * 747796405ull));
        if (state == 0) {
            state = 2891336453u;
        }

        std::uint64_t used = 0;
        mark_card_reference(used, options.hand[0]);
        mark_card_reference(used, options.hand[1]);

        std::array<unsigned char, 5> final_board{};
        for (size_t i = 0; i < options.board.size(); ++i) {
            final_board[i] = options.board[i];
            mark_card_reference(used, options.board[i]);
        }

        std::array<unsigned char, kMaxOpponents * 2> opponents{};
        for (std::uint32_t i = 0; i < options.opponents; ++i) {
            opponents[i * 2] = draw_card_reference(used, state);
            opponents[i * 2 + 1] = draw_card_reference(used, state);
        }

        for (size_t i = options.board.size(); i < final_board.size(); ++i) {
            final_board[i] = draw_card_reference(used, state);
        }

        const std::array<unsigned char, 7> hero_cards = {
            options.hand[0],
            options.hand[1],
            final_board[0],
            final_board[1],
            final_board[2],
            final_board[3],
            final_board[4],
        };
        const std::uint32_t hero_score = eval7_reference(hero_cards);

        std::uint32_t split_count = 1;
        bool beaten = false;
        for (std::uint32_t i = 0; i < options.opponents; ++i) {
            const std::array<unsigned char, 7> villain_cards = {
                opponents[i * 2],
                opponents[i * 2 + 1],
                final_board[0],
                final_board[1],
                final_board[2],
                final_board[3],
                final_board[4],
            };
            const std::uint32_t villain_score = eval7_reference(villain_cards);
            if (villain_score > hero_score) {
                beaten = true;
                break;
            }
            if (villain_score == hero_score) {
                ++split_count;
            }
        }

        if (beaten) {
            ++result.losses;
        } else if (split_count == 1) {
            ++result.wins;
            result.pot_share_total += 1.0;
        } else {
            ++result.ties;
            result.pot_share_total += 1.0 / static_cast<double>(split_count);
        }
    }

    return result;
}

TimingSummary summarize_timings(const std::vector<double>& timings) {
    TimingSummary summary;
    if (timings.empty()) {
        return summary;
    }

    summary.min_ms = timings.front();
    summary.max_ms = timings.front();
    double total = 0.0;
    for (double timing : timings) {
        summary.min_ms = std::min(summary.min_ms, timing);
        summary.max_ms = std::max(summary.max_ms, timing);
        total += timing;
    }
    summary.mean_ms = total / static_cast<double>(timings.size());
    return summary;
}

template <typename Fn>
SimulationResult time_simulation(Fn&& fn, double* elapsed_ms) {
    const auto start = std::chrono::steady_clock::now();
    SimulationResult result = fn();
    const auto end = std::chrono::steady_clock::now();
    *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

double result_equity(const Options& options, const SimulationResult& result) {
    return result.pot_share_total / static_cast<double>(options.trials);
}

Options make_profile_scenario(const Options& base, std::uint64_t scenario_seed) {
    Options scenario = base;
    scenario.seed = scenario_seed;
    scenario.list_devices = false;
    scenario.profile = false;

    std::uint32_t state = static_cast<std::uint32_t>(
        scenario_seed ^ (scenario_seed >> 32) ^ 0x9e3779b9u);
    if (state == 0) {
        state = 2891336453u;
    }

    std::uint64_t used = 0;
    if (base.hand_provided) {
        for (unsigned char card : scenario.hand) {
            mark_card_reference(used, card);
        }
    } else {
        scenario.hand.clear();
    }

    if (base.board_provided) {
        for (unsigned char card : scenario.board) {
            mark_card_reference(used, card);
        }
    } else {
        scenario.board.clear();
    }

    while (scenario.hand.size() < 2) {
        scenario.hand.push_back(draw_card_reference(used, state));
    }

    if (!base.board_provided) {
        const std::uint32_t board_count = next_random_reference(state) % 6u;
        while (scenario.board.size() < board_count) {
            scenario.board.push_back(draw_card_reference(used, state));
        }
    }

    return scenario;
}

void print_timing_summary(const std::string& name, const TimingSummary& summary) {
    std::cout << name
              << " min=" << summary.min_ms << " ms"
              << " mean=" << summary.mean_ms << " ms"
              << " max=" << summary.max_ms << " ms\n";
}

int run_profile(const Options& options) {
    const auto init_start = std::chrono::steady_clock::now();
    OpenCLRuntime runtime = create_opencl_runtime();
    const auto init_end = std::chrono::steady_clock::now();
    const double opencl_init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();

    const std::uint32_t total_runs = options.warmup_runs + options.profile_runs;
    std::vector<Options> scenarios;
    scenarios.reserve(total_runs);
    for (std::uint32_t i = 0; i < total_runs; ++i) {
        const std::uint64_t scenario_seed = options.seed + (static_cast<std::uint64_t>(i + 1) * 0x9e3779b97f4a7c15ull);
        scenarios.push_back(make_profile_scenario(options, scenario_seed));
    }

    for (std::uint32_t i = 0; i < options.warmup_runs; ++i) {
        (void)run_cpu_reference(scenarios[i]);
        (void)run_opencl_simulation(runtime, scenarios[i]);
    }

    std::vector<double> cpu_timings;
    std::vector<double> opencl_timings;
    cpu_timings.reserve(options.profile_runs);
    opencl_timings.reserve(options.profile_runs);

    std::cout << "Profile configuration\n";
    std::cout << "Trials/run:    " << options.trials << '\n';
    std::cout << "Opponents:     " << options.opponents << '\n';
    std::cout << "Warmup runs:   " << options.warmup_runs << '\n';
    std::cout << "Measured runs: " << options.profile_runs << '\n';
    std::cout << "Base seed:     " << options.seed << '\n';
    std::cout << "OpenCL device: " << runtime.device_name << '\n';
    std::cout << "OpenCL init:   " << opencl_init_ms << " ms\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Run  Hand   Board            CPU ms      OpenCL ms   Speedup   Equity\n";

    for (std::uint32_t i = 0; i < options.profile_runs; ++i) {
        const Options& scenario = scenarios[options.warmup_runs + i];

        double cpu_ms = 0.0;
        const SimulationResult cpu_result = time_simulation(
            [&]() { return run_cpu_reference(scenario); },
            &cpu_ms);

        double opencl_ms = 0.0;
        const SimulationResult opencl_result = time_simulation(
            [&]() { return run_opencl_simulation(runtime, scenario); },
            &opencl_ms);

        cpu_timings.push_back(cpu_ms);
        opencl_timings.push_back(opencl_ms);

        const double speedup = opencl_ms > 0.0 ? cpu_ms / opencl_ms : 0.0;
        std::cout << std::setw(3) << (i + 1) << "  "
                  << std::setw(5) << cards_to_string(scenario.hand) << "  "
                  << std::setw(15) << cards_to_string(scenario.board) << "  "
                  << std::setw(10) << cpu_ms << "  "
                  << std::setw(10) << opencl_ms << "  "
                  << std::setw(7) << speedup << "x  "
                  << std::setw(6) << (result_equity(scenario, opencl_result) * 100.0) << "%\n";

        if (cpu_result.wins != opencl_result.wins
            || cpu_result.ties != opencl_result.ties
            || cpu_result.losses != opencl_result.losses
            || cpu_result.pot_share_total != opencl_result.pot_share_total) {
            std::cerr << "warning: CPU reference and OpenCL results differed on run " << (i + 1) << '\n';
        }
    }

    const TimingSummary cpu_summary = summarize_timings(cpu_timings);
    const TimingSummary opencl_summary = summarize_timings(opencl_timings);

    std::cout << "\nSummary\n";
    print_timing_summary("CPU reference:", cpu_summary);
    print_timing_summary("OpenCL hot:   ", opencl_summary);
    if (opencl_summary.mean_ms > 0.0) {
        std::cout << "Mean speedup:  " << (cpu_summary.mean_ms / opencl_summary.mean_ms) << "x\n";
    }

    release_opencl_runtime(runtime);
    return 0;
}

std::string device_type_label(cl_device_type type) {
    if (type == static_cast<cl_device_type>(CL_DEVICE_TYPE_GPU)) {
        return "GPU";
    }
    if (type == static_cast<cl_device_type>(CL_DEVICE_TYPE_DEFAULT)) {
        return "DEFAULT";
    }
    return "UNKNOWN";
}

void print_device_probe(cl_platform_id platform, cl_device_type type) {
    cl_device_id device = nullptr;
    cl_uint count = 0;
    const cl_int count_only_err = clGetDeviceIDs(platform, type, 0, nullptr, &count);
    const cl_int device_only_err = clGetDeviceIDs(platform, type, 1, &device, nullptr);
    const cl_int both_err = clGetDeviceIDs(platform, type, 1, &device, &count);

    std::cout << "  " << device_type_label(type) << ": ";
    if (device_only_err == CL_SUCCESS || both_err == CL_SUCCESS) {
        std::cout << device_info_string(device, CL_DEVICE_NAME) << '\n';
    } else {
        std::cout
            << "count-only=" << cl_error_name(count_only_err)
            << ", device-only=" << cl_error_name(device_only_err)
            << ", both=" << cl_error_name(both_err) << '\n';
    }
}

void print_context_probe(cl_device_type type) {
    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContextFromType(nullptr, type, nullptr, nullptr, &err);

    std::cout << "  " << device_type_label(type) << ": ";
    if (err != CL_SUCCESS) {
        std::cout << cl_error_name(err) << '\n';
        return;
    }

    size_t bytes = 0;
    const cl_int info_err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &bytes);
    if (info_err != CL_SUCCESS) {
        std::cout << "context created, device query failed: " << cl_error_name(info_err) << '\n';
        clReleaseContext(context);
        return;
    }

    std::vector<cl_device_id> devices(bytes / sizeof(cl_device_id));
    check(
        clGetContextInfo(context, CL_CONTEXT_DEVICES, bytes, devices.data(), nullptr),
        "clGetContextInfo diagnostic devices");
    if (devices.empty()) {
        std::cout << "context created, no devices returned\n";
    } else {
        std::cout << device_info_string(devices.front(), CL_DEVICE_NAME) << '\n';
    }
    clReleaseContext(context);
}

int list_opencl_devices() {
    const std::array<cl_device_type, 2> device_order = {
        static_cast<cl_device_type>(CL_DEVICE_TYPE_GPU),
        static_cast<cl_device_type>(CL_DEVICE_TYPE_DEFAULT),
    };

    std::cout << "Default platform probe:\n";
    for (cl_device_type type : device_order) {
        print_device_probe(nullptr, type);
    }

    std::cout << "\nContext-by-type probe:\n";
    for (cl_device_type type : device_order) {
        print_context_probe(type);
    }

    cl_uint platform_count = 0;
    const cl_int platform_count_err = clGetPlatformIDs(0, nullptr, &platform_count);
    std::cout << "\nPlatform count: ";
    if (platform_count_err != CL_SUCCESS) {
        std::cout << cl_error_name(platform_count_err) << '\n';
        return 1;
    }
    std::cout << platform_count << '\n';

    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs list");

    for (size_t i = 0; i < platforms.size(); ++i) {
        std::cout << "\nPlatform " << i << ": "
                  << platform_info_string(platforms[i], CL_PLATFORM_NAME) << '\n';
        for (cl_device_type type : device_order) {
            print_device_probe(platforms[i], type);
        }
    }

    return 0;
}

SimulationResult run_opencl_simulation(OpenCLRuntime& runtime, const Options& options) {
    std::array<unsigned char, 2> hand = {options.hand[0], options.hand[1]};
    std::array<unsigned char, 5> board = {255, 255, 255, 255, 255};
    std::copy(options.board.begin(), options.board.end(), board.begin());

    std::vector<unsigned char> outcomes(static_cast<size_t>(options.trials));
    std::vector<unsigned char> pot_shares(static_cast<size_t>(options.trials));

    cl_mem hand_buffer = make_buffer(
        runtime.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        hand.size() * sizeof(unsigned char),
        hand.data(),
        "hand");
    cl_mem board_buffer = make_buffer(
        runtime.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        board.size() * sizeof(unsigned char),
        board.data(),
        "board");
    cl_mem outcomes_buffer = make_buffer(
        runtime.context,
        CL_MEM_WRITE_ONLY,
        outcomes.size() * sizeof(unsigned char),
        nullptr,
        "outcomes");
    cl_mem pot_shares_buffer = make_buffer(
        runtime.context,
        CL_MEM_WRITE_ONLY,
        pot_shares.size() * sizeof(unsigned char),
        nullptr,
        "pot_shares");

    const cl_uint board_count = static_cast<cl_uint>(options.board.size());
    const cl_uint opponent_count = static_cast<cl_uint>(options.opponents);
    const cl_ulong seed = static_cast<cl_ulong>(options.seed);

    check(clSetKernelArg(runtime.kernel, 0, sizeof(cl_mem), &hand_buffer), "clSetKernelArg hand");
    check(clSetKernelArg(runtime.kernel, 1, sizeof(cl_mem), &board_buffer), "clSetKernelArg board");
    check(clSetKernelArg(runtime.kernel, 2, sizeof(cl_uint), &board_count), "clSetKernelArg board_count");
    check(clSetKernelArg(runtime.kernel, 3, sizeof(cl_uint), &opponent_count), "clSetKernelArg opponent_count");
    check(clSetKernelArg(runtime.kernel, 4, sizeof(cl_ulong), &seed), "clSetKernelArg seed");
    check(clSetKernelArg(runtime.kernel, 5, sizeof(cl_mem), &outcomes_buffer), "clSetKernelArg outcomes");
    check(clSetKernelArg(runtime.kernel, 6, sizeof(cl_mem), &pot_shares_buffer), "clSetKernelArg pot_shares");

    const size_t global_work_size = outcomes.size();
    check(
        clEnqueueNDRangeKernel(runtime.queue, runtime.kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel");
    check(clFinish(runtime.queue), "clFinish");

    check(
        clEnqueueReadBuffer(
            runtime.queue,
            outcomes_buffer,
            CL_TRUE,
            0,
            outcomes.size() * sizeof(unsigned char),
            outcomes.data(),
            0,
            nullptr,
            nullptr),
        "clEnqueueReadBuffer outcomes");
    check(
        clEnqueueReadBuffer(
            runtime.queue,
            pot_shares_buffer,
            CL_TRUE,
            0,
            pot_shares.size() * sizeof(unsigned char),
            pot_shares.data(),
            0,
            nullptr,
            nullptr),
        "clEnqueueReadBuffer pot_shares");

    const SimulationResult result = reduce_outcomes(options, outcomes, pot_shares);

    clReleaseMemObject(pot_shares_buffer);
    clReleaseMemObject(outcomes_buffer);
    clReleaseMemObject(board_buffer);
    clReleaseMemObject(hand_buffer);

    return result;
}

int run_opencl(const Options& options) {
    OpenCLRuntime runtime = create_opencl_runtime();
    const SimulationResult result = run_opencl_simulation(runtime, options);
    print_results(options, result, runtime.device_name);
    release_opencl_runtime(runtime);
    return 0;
}

int run(const Options& options) {
    if (options.list_devices) {
        return list_opencl_devices();
    }
    if (options.profile) {
        return run_profile(options);
    }

    return run_opencl(options);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        return run(options);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
}
