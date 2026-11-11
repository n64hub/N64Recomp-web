#ifndef __RECOMP_H__
#define __RECOMP_H__

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <fenv.h>
#include <assert.h>

// Compiler definition to disable inter-procedural optimization, allowing multiple functions to be in a single file without breaking interposition.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    // MSVC's __declspec(noinline) seems to disable inter-procedural optimization entirely, so it's all that's needed.
    #define RECOMP_FUNC __declspec(noinline)
    
    // Use MSVC's fenv_access pragma.
    #define SET_FENV_ACCESS() _Pragma("fenv_access(on)")
#elif defined(__clang__)
    // Clang has no dedicated IPO attribute, so we use a combination of other attributes to give the desired behavior.
    // The inline keyword allows multiple definitions during linking, and extern forces clang to emit an externally visible definition.
    // Weak forces Clang to not perform any IPO as the symbol can be interposed, which prevents actual inlining due to the inline keyword.
    // Add noinline on for good measure, which doesn't conflict with the inline keyword as they have different meanings.
    #define RECOMP_FUNC extern inline __attribute__((weak,noinline))

    // Use the standard STDC FENV_ACCESS pragma.
    #define SET_FENV_ACCESS() _Pragma("STDC FENV_ACCESS ON")
#elif defined(__GNUC__) && !defined(__INTEL_COMPILER)
    // Use GCC's attribute for disabling inter-procedural optimizations. Also enable the rounding-math compiler flag to disable
    // constant folding so that arithmetic respects the floating point environment. This is needed because gcc doesn't implement
    // any FENV_ACCESS pragma.
    #define RECOMP_FUNC __attribute__((noipa, optimize("rounding-math")))

    // There's no FENV_ACCESS pragma in gcc, so this can be empty.
    #define SET_FENV_ACCESS()
#else
    #error "No RECOMP_FUNC definition for this compiler"
#endif

// Implementation of 64-bit multiply and divide instructions
#if defined(__SIZEOF_INT128__)

static inline void DMULT(int64_t a, int64_t b, int64_t* lo64, int64_t* hi64) {
    __int128 full128 = ((__int128)a) * ((__int128)b);

    *hi64 = (int64_t)(full128 >> 64);
    *lo64 = (int64_t)(full128 >> 0);
}

static inline void DMULTU(uint64_t a, uint64_t b, uint64_t* lo64, uint64_t* hi64) {
    unsigned __int128 full128 = ((unsigned __int128)a) * ((unsigned __int128)b);

    *hi64 = (uint64_t)(full128 >> 64);
    *lo64 = (uint64_t)(full128 >> 0);
}

#elif defined(_MSC_VER)

#include <intrin.h>
#pragma intrinsic(_mul128)
#pragma intrinsic(_umul128)

static inline void DMULT(int64_t a, int64_t b, int64_t* lo64, int64_t* hi64) {
    *lo64 = _mul128(a, b, hi64);
}

static inline void DMULTU(uint64_t a, uint64_t b, uint64_t* lo64, uint64_t* hi64) {
    *lo64 = _umul128(a, b, hi64);
}

#else
#error "128-bit integer type not found"
#endif

static inline void DDIV(int64_t a, int64_t b, int64_t* quot, int64_t* rem) {
    int overflow = ((uint64_t)a == 0x8000000000000000ull) && (b == -1ll);
    *quot = overflow ? a : (a / b);
    *rem = overflow ? 0 : (a % b);
}

static inline void DDIVU(uint64_t a, uint64_t b, uint64_t* quot, uint64_t* rem) {
    *quot = a / b;
    *rem = a % b;
}

typedef uint64_t gpr;

#define SIGNED(val) \
    ((int64_t)(val))

#define ADD32(a, b) \
    ((gpr)(int32_t)((a) + (b)))

#define SUB32(a, b) \
    ((gpr)(int32_t)((a) - (b)))

// Where an address lies from rdram: KSEG0 (0x80000000 up) at its offset, and the rest 2 GB or more past it, which a
// native build reserves (librecomp's allocation_size) and a recomp whose code reads through TLB-mapped KUSEG addresses
// opens as it needs (recomp::make_kuseg_accessible). wasm32's pointers are 32 bits, so there rdram + 0x80000000 and up
// wraps to anywhere in the heap. A recomp that defines RECOMP_KUSEG_WINDOW for every target it builds (the runtime's
// too) gets a window for KUSEG's first 16 MB right after kseg0's 512 MB instead, which librecomp allocates with
// RDRAM; a KUSEG address past it wraps inside it, never into the heap. ultramodern's TO_PTR maps the same way.
// Without it, the offset is the native one.
#if defined(__wasm__) && defined(RECOMP_KUSEG_WINDOW)
#define RECOMP_KUSEG_WINDOW_BASE 0x20000000u
#define RECOMP_KUSEG_WINDOW_SIZE 0x01000000u

static inline uint32_t recomp_rdram_offset(gpr address) {
    uint32_t a = (uint32_t)address;
    return (a & 0x80000000u) ? a - 0x80000000u : RECOMP_KUSEG_WINDOW_BASE + (a & (RECOMP_KUSEG_WINDOW_SIZE - 1));
}

#define RECOMP_RDRAM_OFFSET(address) recomp_rdram_offset(address)
#else
#define RECOMP_RDRAM_OFFSET(address) ((address) - 0xFFFFFFFF80000000)
#endif

#define MEM_W(offset, reg) \
    (*(int32_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset)))))

#define MEM_H(offset, reg) \
    (*(int16_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset)) ^ 2)))

#define MEM_B(offset, reg) \
    (*(int8_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset)) ^ 3)))

#define MEM_HU(offset, reg) \
    (*(uint16_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset)) ^ 2)))

#define MEM_BU(offset, reg) \
    (*(uint8_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset)) ^ 3)))

#define SD(val, offset, reg) { \
    *(uint32_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset) + 4))) = (uint32_t)((gpr)(val) >> 0); \
    *(uint32_t*)(rdram + RECOMP_RDRAM_OFFSET(((reg) + (offset) + 0))) = (uint32_t)((gpr)(val) >> 32); \
}

static inline uint64_t load_doubleword(uint8_t* rdram, gpr reg, gpr offset) {
    uint64_t ret = 0;
    uint64_t lo = (uint64_t)(uint32_t)MEM_W(reg, offset + 4);
    uint64_t hi = (uint64_t)(uint32_t)MEM_W(reg, offset + 0);
    ret = (lo << 0) | (hi << 32);
    return ret;
}

#define LD(offset, reg) \
    load_doubleword(rdram, offset, reg)

static inline gpr do_lwl(uint8_t* rdram, gpr initial_value, gpr offset, gpr reg) {
    // Calculate the overall address
    gpr address = (offset + reg);

    // Load the aligned word
    gpr word_address = address & ~0x3;
    uint32_t loaded_value = MEM_W(0, word_address);

    // Mask the existing value and shift the loaded value appropriately
    gpr misalignment = address & 0x3;
    gpr masked_value = initial_value & (gpr)(uint32_t)~(0xFFFFFFFFu << (misalignment * 8));
    loaded_value <<= (misalignment * 8);

    // Cast to int32_t to sign extend first
    return (gpr)(int32_t)(masked_value | loaded_value);
}

static inline gpr do_lwr(uint8_t* rdram, gpr initial_value, gpr offset, gpr reg) {
    // Calculate the overall address
    gpr address = (offset + reg);
    
    // Load the aligned word
    gpr word_address = address & ~0x3;
    uint32_t loaded_value = MEM_W(0, word_address);

    // Mask the existing value and shift the loaded value appropriately
    gpr misalignment = address & 0x3;
    gpr masked_value = initial_value & (gpr)(uint32_t)~(0xFFFFFFFFu >> (24 - misalignment * 8));
    loaded_value >>= (24 - misalignment * 8);

    // Cast to int32_t to sign extend first
    return (gpr)(int32_t)(masked_value | loaded_value);
}

static inline void do_swl(uint8_t* rdram, gpr offset, gpr reg, gpr val) {
    // Calculate the overall address
    gpr address = (offset + reg);

    // Get the initial value of the aligned word
    gpr word_address = address & ~0x3;
    uint32_t initial_value = MEM_W(0, word_address);

    // Mask the initial value and shift the input value appropriately
    gpr misalignment = address & 0x3;
    uint32_t masked_initial_value = initial_value & ~(0xFFFFFFFFu >> (misalignment * 8));
    uint32_t shifted_input_value = ((uint32_t)val) >> (misalignment * 8);
    MEM_W(0, word_address) = masked_initial_value | shifted_input_value;
}

static inline void do_swr(uint8_t* rdram, gpr offset, gpr reg, gpr val) {
    // Calculate the overall address
    gpr address = (offset + reg);

    // Get the initial value of the aligned word
    gpr word_address = address & ~0x3;
    uint32_t initial_value = MEM_W(0, word_address);

    // Mask the initial value and shift the input value appropriately
    gpr misalignment = address & 0x3;
    uint32_t masked_initial_value = initial_value & ~(0xFFFFFFFFu << (24 - misalignment * 8));
    uint32_t shifted_input_value = ((uint32_t)val) << (24 - misalignment * 8);
    MEM_W(0, word_address) = masked_initial_value | shifted_input_value;
}

static inline gpr do_ldl(uint8_t* rdram, gpr initial_value, gpr offset, gpr reg) {
    // Calculate the overall address
    gpr address = (offset + reg);

    // Load the aligned dword
    gpr dword_address = address & ~0x7;
    uint64_t loaded_value = load_doubleword(rdram, 0, dword_address);

    // Mask the existing value and shift the loaded value appropriately
    gpr misalignment = address & 0x7;
    gpr masked_value = initial_value & ~(0xFFFFFFFFFFFFFFFFu << (misalignment * 8));
    loaded_value <<= (misalignment * 8);

    return masked_value | loaded_value;
}

static inline gpr do_ldr(uint8_t* rdram, gpr initial_value, gpr offset, gpr reg) {
    // Calculate the overall address
    gpr address = (offset + reg);
    
    // Load the aligned dword
    gpr dword_address = address & ~0x7;
    uint64_t loaded_value = load_doubleword(rdram, 0, dword_address);

    // Mask the existing value and shift the loaded value appropriately
    gpr misalignment = address & 0x7;
    gpr masked_value = initial_value & ~(0xFFFFFFFFFFFFFFFFu >> (56 - misalignment * 8));
    loaded_value >>= (56 - misalignment * 8);

    return masked_value | loaded_value;
}

static inline void do_sdl(uint8_t* rdram, gpr offset, gpr reg, gpr val) {
    // Calculate the overall address
    gpr address = (offset + reg);

    // Get the initial value of the aligned dword
    gpr dword_address = address & ~0x7;
    uint64_t initial_value = load_doubleword(rdram, 0, dword_address);

    // Mask the initial value and shift the input value appropriately
    gpr misalignment = address & 0x7;
    uint64_t masked_initial_value = initial_value & ~(0xFFFFFFFFFFFFFFFFu >> (misalignment * 8));
    uint64_t shifted_input_value = val >> (misalignment * 8);

    uint64_t ret = masked_initial_value | shifted_input_value;
    uint32_t lo = (uint32_t)ret;
    uint32_t hi = (uint32_t)(ret >> 32);

    MEM_W(0, dword_address + 4) = lo;
    MEM_W(0, dword_address + 0) = hi;
}

static inline void do_sdr(uint8_t* rdram, gpr offset, gpr reg, gpr val) {
    // Calculate the overall address
    gpr address = (offset + reg);

    // Get the initial value of the aligned dword
    gpr dword_address = address & ~0x7;
    uint64_t initial_value = load_doubleword(rdram, 0, dword_address);

    // Mask the initial value and shift the input value appropriately
    gpr misalignment = address & 0x7;
    uint64_t masked_initial_value = initial_value & ~(0xFFFFFFFFFFFFFFFFu << (56 - misalignment * 8));
    uint64_t shifted_input_value = val << (56 - misalignment * 8);
    
    uint64_t ret = masked_initial_value | shifted_input_value;
    uint32_t lo = (uint32_t)ret;
    uint32_t hi = (uint32_t)(ret >> 32);

    MEM_W(0, dword_address + 4) = lo;
    MEM_W(0, dword_address + 0) = hi;
}

#if defined(__wasm__)
// WebAssembly only rounds to nearest, and its fesetround accepts every mode and changes nothing. So the mode the game
// sets is kept here, per thread as the host's would be, and the conversions that take the current mode (cvt.w, cvt.l)
// apply it themselves. Arithmetic still rounds to nearest whatever the mode.
#ifdef __cplusplus
#define RECOMP_THREAD_LOCAL thread_local
#else
#define RECOMP_THREAD_LOCAL _Thread_local
#endif
__attribute__((weak)) RECOMP_THREAD_LOCAL uint32_t recomp_rounding_mode = 0;

static inline uint32_t get_cop1_cs() {
    return recomp_rounding_mode;
}

static inline void set_cop1_cs(uint32_t val) {
    recomp_rounding_mode = val & 0x3;
}

static inline float recomp_round_s(float val) {
    switch (recomp_rounding_mode) {
        case 1: return truncf(val);
        case 2: return ceilf(val);
        case 3: return floorf(val);
        default: return rintf(val);
    }
}

static inline double recomp_round_d(double val) {
    switch (recomp_rounding_mode) {
        case 1: return trunc(val);
        case 2: return ceil(val);
        case 3: return floor(val);
        default: return rint(val);
    }
}
#else
static inline uint32_t get_cop1_cs() {
    uint32_t rounding_mode = 0;
    switch (fegetround()) {
        // round to nearest value
        case FE_TONEAREST:
        default:
            rounding_mode = 0;
            break;
        // round to zero (truncate)
        case FE_TOWARDZERO:
            rounding_mode = 1;
            break;
        // round to positive infinity (ceil)
        case FE_UPWARD:
            rounding_mode = 2;
            break;
        // round to negative infinity (floor)
        case FE_DOWNWARD:
            rounding_mode = 3;
            break;
    }
    return rounding_mode;
}

static inline void set_cop1_cs(uint32_t val) {
    uint32_t rounding_mode = val & 0x3;
    int round = FE_TONEAREST;
    switch (rounding_mode) {
        case 0: // round to nearest value
            round = FE_TONEAREST;
            break;
        case 1: // round to zero (truncate)
            round = FE_TOWARDZERO;
            break;
        case 2: // round to positive infinity (ceil)
            round = FE_UPWARD;
            break;
        case 3: // round to negative infinity (floor)
            round = FE_DOWNWARD;
            break;
    }
    fesetround(round);
}
#endif

#define S32(val) \
    ((int32_t)(val))
    
#define U32(val) \
    ((uint32_t)(val))

#define S64(val) \
    ((int64_t)(val))

#define U64(val) \
    ((uint64_t)(val))

#define MUL_S(val1, val2) \
    ((val1) * (val2))

#define MUL_D(val1, val2) \
    ((val1) * (val2))

#define DIV_S(val1, val2) \
    ((val1) / (val2))

#define DIV_D(val1, val2) \
    ((val1) / (val2))

#define CVT_S_W(val) \
    ((float)((int32_t)(val)))

#define CVT_D_W(val) \
    ((double)((int32_t)(val)))

#define CVT_D_L(val) \
    ((double)((int64_t)(val)))

#define CVT_S_L(val) \
    ((float)((int64_t)(val)))

#define CVT_D_S(val) \
    ((double)(val))

#define CVT_S_D(val) \
    ((float)(val))

#define TRUNC_W_S(val) \
    ((int32_t)(val))

#define TRUNC_W_D(val) \
    ((int32_t)(val))

#define TRUNC_L_S(val) \
    ((int64_t)(val))

#define TRUNC_L_D(val) \
    ((int64_t)(val))

#define DEFAULT_ROUNDING_MODE 0

static inline int32_t do_cvt_w_s(float val) {
    // Rounding mode aware float to 32-bit int conversion.
#if defined(__wasm__)
    return (int32_t)recomp_round_s(val);
#else
    return (int32_t)lrintf(val);
#endif
}

#define CVT_W_S(val) \
    do_cvt_w_s(val)

static inline int64_t do_cvt_l_s(float val) {
    // Rounding mode aware float to 64-bit int conversion.
#if defined(__wasm__)
    return (int64_t)recomp_round_s(val);
#else
    return (int64_t)llrintf(val);
#endif
}

#define CVT_L_S(val) \
    do_cvt_l_s(val);

static inline int32_t do_cvt_w_d(double val) {
    // Rounding mode aware double to 32-bit int conversion.
#if defined(__wasm__)
    return (int32_t)recomp_round_d(val);
#else
    return (int32_t)lrint(val);
#endif
}

#define CVT_W_D(val) \
    do_cvt_w_d(val)

static inline int64_t do_cvt_l_d(double val) {
    // Rounding mode aware double to 64-bit int conversion.
#if defined(__wasm__)
    return (int64_t)recomp_round_d(val);
#else
    return (int64_t)llrint(val);
#endif
}

#define CVT_L_D(val) \
    do_cvt_l_d(val)

#define NAN_CHECK(val) \
    assert(val == val)

//#define NAN_CHECK(val)

typedef union {
    double d;
    struct {
        float fl;
        float fh;
    };
    struct {
        uint32_t u32l;
        uint32_t u32h;
    };
    uint64_t u64;
} fpr;

typedef struct {
    gpr r0,  r1,  r2,  r3,  r4,  r5,  r6,  r7,
        r8,  r9,  r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23,
        r24, r25, r26, r27, r28, r29, r30, r31;
    fpr f0,  f1,  f2,  f3,  f4,  f5,  f6,  f7,
        f8,  f9,  f10, f11, f12, f13, f14, f15,
        f16, f17, f18, f19, f20, f21, f22, f23,
        f24, f25, f26, f27, f28, f29, f30, f31;
    uint64_t hi, lo;
    uint32_t* f_odd;
    uint32_t status_reg;
    uint8_t mips3_float_mode;
} recomp_context;

// Checks if the target is an even float register or that mips3 float mode is enabled
#define CHECK_FR(ctx, idx) \
    assert(((idx) & 1) == 0 || (ctx)->mips3_float_mode)

#ifdef __cplusplus
extern "C" {
#endif

void cop0_status_write(recomp_context* ctx, gpr value);
gpr cop0_status_read(recomp_context* ctx);
void switch_error(const char* func, uint32_t vram, uint32_t jtbl);
void do_break(uint32_t vram);

// The function signature for all recompiler output functions.
typedef void (recomp_func_t)(uint8_t* rdram, recomp_context* ctx);
// The function signature for special functions that need a third argument.
// These get called via generated shims to allow providing some information about the caller, such as mod id.
typedef void (recomp_func_ext_t)(uint8_t* rdram, recomp_context* ctx, uintptr_t arg);

recomp_func_t* get_function(int32_t vram);

#define LOOKUP_FUNC(val) \
    get_function((int32_t)(val))

extern int32_t* section_addresses;

#define LO16(x) \
    ((x) & 0xFFFF)

#define HI16(x) \
    (((x) >> 16) + (((x) >> 15) & 1))

#define RELOC_HI16(section_index, offset) \
    HI16(section_addresses[section_index] + (offset))

#define RELOC_LO16(section_index, offset) \
    LO16(section_addresses[section_index] + (offset))

void recomp_syscall_handler(uint8_t* rdram, recomp_context* ctx, int32_t instruction_vram);

void pause_self(uint8_t *rdram);

#if defined(__wasm__)
// A mod replaces or hooks a recompiled function by rewriting its machine code (N64ModernRuntime's patch_func), which
// WebAssembly can't do, so there every recompiled function asks at its entry instead. A function pointer is an index
// into the function table, and recomp_redirected has a byte for each index below recomp_redirected_count, set for a
// function a mod replaces or hooks; recomp_redirect then runs the replacement or the hooks around the function and
// returns nonzero, or returns 0 when the function is to run itself (as it is called from those hooks). Without mods
// the count is 0, and the check is one comparison.
extern const uint8_t* recomp_redirected;
extern uintptr_t recomp_redirected_count;
int recomp_redirect(uint8_t* rdram, recomp_context* ctx, uintptr_t func);
// A function's table index, hidden from the optimizer: clang (24) folds a function's address used as an array index
// into the load's offset, and the linker fills that in with the function's index in the module, not its table index.
static inline uintptr_t recomp_table_index(uintptr_t func) {
    __asm__("" : "+r"(func));
    return func;
}
#define RECOMP_FUNC_ENTRY(func) \
    { \
        uintptr_t recomp_entry_index = recomp_table_index((uintptr_t)&func); \
        if (recomp_entry_index < recomp_redirected_count && recomp_redirected[recomp_entry_index] && \
            recomp_redirect(rdram, ctx, recomp_entry_index)) return; \
    }
#else
#define RECOMP_FUNC_ENTRY(func)
#endif

// What comes before a tail call, a call to another function as this one's last act (a MIPS j or jr out of it), which
// the recompiled C writes as the call and a return. Natively the optimizer turns that into a jump. In WebAssembly it
// stays a call, so a game whose code chains tail calls without end overflows the browser's stack (Conker's Bad Fur
// Day after minutes of play), unless it is a return_call: with the tail call feature (-mtail-call, which every
// browser with WebGPU has) this forces one. Without it the call stays a plain one. The empty statement keeps the
// attribute off a label just before it, which would take it as its own.
#if defined(__wasm_tail_call__)
#define RECOMP_TAIL_CALL ; __attribute__((musttail)) return
#else
#define RECOMP_TAIL_CALL
#endif

// A mod's code recompiled ahead of time into C (OfflineModRecomp --static) and linked into the executable, for a
// platform without the live recompiler (WebAssembly). The runtime uses it, through the pointers here, for a mod whose
// mod_binary.bin and mod_syms.bin hash to `hash` (N64Recomp::mod_code_hash), in place of recompiling the mod.
typedef struct {
    uint64_t hash;
    size_t num_functions;
    recomp_func_t* const* functions;      // in the order of the mod's symbol file
    recomp_func_t** imported_funcs;
    recomp_func_t** reference_symbol_funcs;
    uint32_t* base_event_index;
    int32_t** reference_section_addresses;
    int32_t* section_addresses;           // the mod's own
} recomp_precompiled_mod;

#ifdef __cplusplus
}
#endif

#endif
