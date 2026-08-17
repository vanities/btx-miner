cmake_minimum_required(VERSION 3.20)

# Backend is selected up front so we can enable_language(CUDA) before project().
option(MATADOR_ENABLE_CUDA "Build the CUDA backend (NVIDIA, sm_120 default)" OFF)
set(MATADOR_CUDA_ARCH "120" CACHE STRING "CUDA SM architectures (e.g. 120 or 80;86;89;90;120)")

if(MATADOR_ENABLE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${MATADOR_CUDA_ARCH})
  project(matador_core CXX CUDA)
else()
  project(matador_core CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- byte-exact PoW core (CPU reference + backend abstraction) ---
file(GLOB CORE_MATMUL ${CMAKE_CURRENT_SOURCE_DIR}/matmul/*.cpp)

# --- minimal vendored btx utilities (NOT the node) ---
file(GLOB_RECURSE CORE_VENDOR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/*.cpp)

# Enable the SHA-NI / AVX2 / SSE4.1 SHA-256 paths in the vendored sha256.cpp (x86_64 only).
# btx's sha256.cpp ships sha256_x86_shani::Transform etc. but they're #if-gated on ENABLE_*
# and need -msha/-mavx2 codegen. With these on, SHA256AutoDetect() (called at miner startup)
# upgrades CSHA256::Transform to SHA-NI -- ~3.6x single-buffer on Zen, used by the E4 CPU
# pre-hash scanner. Byte-identical output (SHA-256 is SHA-256; the digest gate still holds).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # Dispatcher (sha256.cpp): just the ENABLE_* defines so SHA256AutoDetect wires up the fast
  # Transforms. It only holds function pointers/forward-decls -- no intrinsics -- so no ISA flag.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vendor/crypto/sha256.cpp PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_SSE41;ENABLE_X86_SHANI;ENABLE_AVX2")
  # Impl TUs (vendored from btx): each provides one Transform and needs its ISA codegen flag;
  # the SHA-NI file is also #if-gated on the ENABLE_* defines. sha256_sse4.cpp is x86_64 inline
  # asm (no flag). These resolve the sha256_sse4/sse41/avx2/x86_shani::Transform refs.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vendor/crypto/sha256_sse41.cpp PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_SSE41"
    COMPILE_OPTIONS "-msse4.1")
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vendor/crypto/sha256_avx2.cpp PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_AVX2"
    COMPILE_OPTIONS "-mavx;-mavx2")
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vendor/crypto/sha256_x86_shani.cpp PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_SSE41;ENABLE_X86_SHANI"
    COMPILE_OPTIONS "-msse4.1;-msha")
endif()

# Same idea for arm64/aarch64 (the macos-arm64 asset + any aarch64 Linux build): the vendored
# sha256.cpp already carries the full ENABLE_ARM_SHANI runtime-dispatch path (getauxval HWCAP_SHA2
# on Linux, the hw.optional.arm.FEAT_SHA256 sysctl on Apple) -- it just needs the define plus the
# impl TU. sha256_arm_shani.cpp (vendored from btx, byte-identical to upstream Bitcoin Core) has
# the vsha256* intrinsics, so that TU alone gets +crypto codegen; the dispatcher holds only
# function pointers/forward-decls (no ISA flag). SHA-256 output is byte-identical (digest gate +
# SHA256AutoDetect's SelfTest both hold) -- this only upgrades CSHA256 to the SHA2 unit (~4x).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vendor/crypto/sha256.cpp PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_ARM_SHANI")
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/vendor/crypto/sha256_arm_shani.cpp PROPERTIES
    COMPILE_DEFINITIONS "ENABLE_ARM_SHANI"
    COMPILE_OPTIONS "-march=armv8-a+crypto")
endif()

# --- backend sources (the ENC_RC accelerators; CPU oracle needs none) ---
set(CORE_BACKEND)

if(MATADOR_ENABLE_CUDA)
  list(APPEND CORE_BACKEND
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rc_episode_accel.cu   # GPU ENC_RC episode backend (byte-exact)
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rc_coupled_accel.cu)  # GPU ENC_RC coupled-V3 backend (byte-exact, device-derived bank)
  # matmul_accel.cu dlopen/dlsym's the green-context driver entry points instead of hard-linking
  # them (they're CUDA 12.4+ only; hard-linking makes the whole binary refuse to load on older
  # drivers even though that path is opt-in). CMAKE_DL_LIBS is empty on glibc >=2.34 (dl folded
  # into libc) and non-empty on older glibc/musl -- always link it, it's a no-op when empty.
  set(CORE_DL_LIBS ${CMAKE_DL_LIBS})
endif()

# ---------------------------------------------------------------------------
# cuBLASLt: prune to the architectures we actually ship.
#
# Stock libcublasLt_static.a is 799 MB (CUDA 13.3) and carries SASS for every arch
# NVIDIA supports. Measured in the v0.9.2 release binary: 2,785 cubins across 11
# archs, of which 1,219 (44%) are sm_100/sm_100a/sm_103 datacenter Blackwell plus
# the retired sm_75 Turing lane -- hardware this binary cannot run on, since we
# build for MATADOR_CUDA_ARCH only. nvprune strips them.
#
# Measured 2026-08-05, relinking the exact v0.9.2 objects:
#   binary  598,337,464 -> 516,630,456 bytes  (-13.7%)
#   HiveOS package  403 MB -> 322 MB          (-20%)
#
# This removes DEVICE CODE ONLY, for archs we never emit our own kernels for, so no
# instruction that ever executes changes. The digest gate is unaffected.
#
# GOTCHA: nvprune's --arch does NOT accumulate. Passing it repeatedly warns
# "incompatible redefinition for option 'arch'" and silently keeps only the LAST
# value -- pruning a multi-arch build down to one arch. Multiple archs REQUIRE
# repeated -gencode. (Cost a wrong measurement 2026-08-05.)
#
# Pruning runs at configure time: it is a ~1 s operation on a toolkit file that never
# changes, and a configure-time file dodges the custom-target ordering problem of a
# generated archive consumed by several targets via a PUBLIC link.
option(MATADOR_PRUNE_CUBLASLT "nvprune libcublasLt_static.a down to MATADOR_CUDA_ARCH" ON)

function(_matador_cublaslt_lib out_var)
  set(${out_var} "CUDA::cublasLt_static" PARENT_SCOPE)   # fallback: stock, unpruned
  if(NOT MATADOR_PRUNE_CUBLASLT)
    return()
  endif()
  find_program(MATADOR_NVPRUNE nvprune HINTS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_ROOT}/bin")
  set(_src "${CUDAToolkit_LIBRARY_DIR}/libcublasLt_static.a")
  if(NOT MATADOR_NVPRUNE OR NOT EXISTS "${_src}")
    message(WARNING "matador: nvprune or libcublasLt_static.a not found -- linking the "
                    "stock (unpruned) cuBLASLt; the binary will be ~80 MB larger")
    return()
  endif()
  set(_gencode "")
  foreach(_a IN LISTS MATADOR_CUDA_ARCH)
    string(REGEX REPLACE "[^0-9]" "" _n "${_a}")         # 90a -> 90
    if(_n)
      list(APPEND _gencode -gencode "arch=compute_${_n},code=sm_${_n}")
    endif()
  endforeach()
  if(NOT _gencode)
    return()
  endif()
  set(_out "${CMAKE_BINARY_DIR}/libcublasLt_static_pruned.a")
  # Re-prune only when the arch set or the source archive changes.
  set(_key "${MATADOR_CUDA_ARCH}")
  if(NOT EXISTS "${_out}" OR NOT "${MATADOR_CUBLASLT_PRUNED_KEY}" STREQUAL "${_key}"
     OR "${_src}" IS_NEWER_THAN "${_out}")
    message(STATUS "matador: pruning cuBLASLt to sm_[${MATADOR_CUDA_ARCH}] (this takes a moment)")
    execute_process(COMMAND "${MATADOR_NVPRUNE}" ${_gencode} "${_src}" -o "${_out}"
                    RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
      message(WARNING "matador: nvprune failed (${_err}) -- linking stock cuBLASLt")
      file(REMOVE "${_out}")
      return()
    endif()
    set(MATADOR_CUBLASLT_PRUNED_KEY "${_key}" CACHE INTERNAL "arch set the pruned cuBLASLt was built for")
  endif()
  file(SIZE "${_out}" _sz)
  math(EXPR _mb "${_sz} / 1000000")
  message(STATUS "matador: pruned cuBLASLt = ${_mb} MB (stock is 799 MB)")
  # Wrap the pruned archive in an IMPORTED target that inherits CUDA::cublasLt_static's
  # INTERFACE_LINK_LIBRARIES rather than linking the bare .a path. cuBLASLt's objects call
  # into culibos (culibosLoadLibrary/culibosGetProcAddress); that dependency lives on the
  # imported target, and a raw file path drops it -- the miner then fails to link with
  # "undefined reference to culibosGetProcAddress" even though rc_probe builds fine.
  if(NOT TARGET matador::cublasLt_pruned)
    add_library(matador::cublasLt_pruned STATIC IMPORTED GLOBAL)
    set_target_properties(matador::cublasLt_pruned PROPERTIES IMPORTED_LOCATION "${_out}")
    get_target_property(_lt_iface CUDA::cublasLt_static INTERFACE_LINK_LIBRARIES)
    if(_lt_iface)
      set_target_properties(matador::cublasLt_pruned PROPERTIES
                            INTERFACE_LINK_LIBRARIES "${_lt_iface}")
    endif()
  endif()
  set(${out_var} "matador::cublasLt_pruned" PARENT_SCOPE)
endfunction()

add_library(matador_core STATIC ${CORE_MATMUL} ${CORE_VENDOR} ${CORE_BACKEND})

if(MATADOR_ENABLE_CUDA)
  target_link_libraries(matador_core PUBLIC ${CORE_DL_LIBS})
  # rc_episode_accel.cu (the GPU ENC_RC episode backend) uses cuBLASLt int8 IMMA + the CUDA
  # runtime. MATADOR_ENABLE_CUDA is exported so callers can compile-guard the GPU episode path.
  find_package(CUDAToolkit REQUIRED)
  # cudart+cublasLt: the RC episode backend (rc_episode_accel.cu). cuda_driver: matmul_accel.cu's
  # driver-API calls (cuDeviceGet, green-context). PUBLIC so every consumer (miner + probes)
  # resolves them regardless of link order.
  # STATIC cudart+cublasLt (2026-08-02): mining hosts have the DRIVER but not the toolkit --
  # dynamic libcudart/libcublasLt crash-looped the v0.9.0 rig on load ("libcudart.so.13: cannot
  # open shared object file"). libcuda stays dynamic: it ships with the driver, always present.
  target_link_libraries(matador_core PUBLIC CUDA::cudart_static CUDA::cuda_driver)
  target_compile_definitions(matador_core PUBLIC MATADOR_ENABLE_CUDA)
  # cuBLASLt is linked ONLY when it is actually the GEMM backend. With CUTLASS present the
  # solvers compile it out (MATADOR_USE_CUBLASLT=0 in rc_gpu_episode.cu) and the ~530 MB of
  # precompiled kernel catalog goes with it. MATADOR_USE_CUBLASLT=ON forces both backends into
  # one binary for A/B work -- at full size.
  if(MATADOR_USE_CUBLASLT OR NOT (MATADOR_CUTLASS_DIR AND EXISTS "${MATADOR_CUTLASS_DIR}/include/cutlass/cutlass.h"))
    _matador_cublaslt_lib(_matador_ltlib)
    target_link_libraries(matador_core PUBLIC ${_matador_ltlib})
    if(MATADOR_USE_CUBLASLT)
      target_compile_definitions(matador_core PUBLIC MATADOR_USE_CUBLASLT=1)
    endif()
    message(STATUS "matador: GEMM backend = cuBLASLt (linked)")
  else()
    message(STATUS "matador: GEMM backend = CUTLASS (cuBLASLt NOT linked)")
  endif()
  # CUTLASS is header-only -- an include path, no library and no runtime dependency. It gates
  # the BTX_RC_GEMM_BACKEND=cutlass path in rc_gpu_episode.cu; without it that path compiles out
  # entirely and cuBLASLt stays the only backend.
  if(MATADOR_CUTLASS_DIR AND EXISTS "${MATADOR_CUTLASS_DIR}/include/cutlass/cutlass.h")
    target_include_directories(matador_core PUBLIC "${MATADOR_CUTLASS_DIR}/include")
    target_compile_definitions(matador_core PUBLIC MATADOR_HAVE_CUTLASS=1)
  endif()
endif()

target_include_directories(matador_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}          # <matmul/...>, <cuda/...>
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor          # <uint256.h>, <crypto/...>, <primitives/...>, shims
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/univalue/include)  # <univalue.h> (vendored univalue for the miner's JSON/GBT)

# Accelerate/BLAS is Apple-only: the sole cblas_dgemm caller (matmul_pow.cpp's
# LowRankProductAccelerateSplit16 AMX path) is entirely inside #if defined(__APPLE__);
# off-Apple the scalar reference runs and no BLAS symbol is referenced anywhere in the
# tree (verified: grep core/ for cblas_/dgemm), so Linux builds need no BLAS at all.
if(APPLE)
  target_link_libraries(matador_core PUBLIC "-framework Accelerate")
endif()

# ---------------------------------------------------------------------------
# Probe / unit-test helpers. Every harness/<name>.cpp follows one of two shapes,
# so the add_executable + link boilerplate is defined once instead of per target.
#
#   matador_probe(<name> [CUDA] [THREADS]) -- links matador_core + btx_log_stub
#       (byte-exact CPU-reference / GPU-checksum harnesses). CUDA also links
#       CUDA::cuda_driver and only builds when MATADOR_ENABLE_CUDA.
#   matador_unit_test(<name>)              -- standalone, dependency-free
#       (the .cpp includes ../miner/<x>.h directly; exit 0 = pass).
# Each is a no-op when its harness/<name>.cpp is absent.
# ---------------------------------------------------------------------------
function(matador_probe name)
  cmake_parse_arguments(P "CUDA;THREADS" "" "" ${ARGN})
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/../harness/${name}.cpp)
  if(NOT EXISTS ${src})
    return()
  endif()
  if(P_CUDA AND NOT MATADOR_ENABLE_CUDA)
    return()
  endif()
  add_executable(${name} ${src} ${CMAKE_CURRENT_SOURCE_DIR}/../harness/btx_log_stub.cpp)
  target_link_libraries(${name} PRIVATE matador_core)
  if(P_THREADS)
    find_package(Threads REQUIRED)
    target_link_libraries(${name} PRIVATE Threads::Threads)
  endif()
  if(P_CUDA)
    target_link_libraries(${name} PRIVATE CUDA::cuda_driver)
  endif()
endfunction()

function(matador_unit_test name)
  cmake_parse_arguments(U "CORE" "" "" ${ARGN})
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/../harness/${name}.cpp)
  if(EXISTS ${src})
    add_executable(${name} ${src})
    if(U_CORE)
      target_link_libraries(${name} PRIVATE matador_core)   # arith_uint256 + vendor include dirs
    endif()
  endif()
endfunction()

# rc_probe is THE byte-exact regression gate: it must print the ENC_RC episode
# golden 5b1bff3c. (It replaced digest_probe's 1dad86f1 when v3 was removed --
# that digest describes a solver this build no longer contains.) The GPU probes
# gate the accelerated backends against the same goldens:
#   cmake --build <build> --target rc_probe && ./<build>/rc_probe
matador_probe(rc_probe)   # v4.6 ENC_RC episode golden gate: 5b1bff3c (inert until nMatMulRCHeight; single-switch activation upstream)
matador_probe(rc_gpu_accel_probe CUDA)  # GPU episode backend byte-exact gate: GPU digest == CPU oracle == golden 5b1bff3c
matador_probe(rc_attest CUDA METAL_STUB)

# rc_attest signs with btx's OWN CKey/ECC so the bytes match what the node verifies.
# That pulls in key/lockedpool/secp256k1, which the generic probe link set omits.
if(TARGET rc_attest)
  # rc_attest signs with btx's OWN CKey/ECC so the emitted bytes are exactly what the
  # node verifies. Same archive set share_verify_probe uses (key/lockedpool/univalue/secp).
  target_link_libraries(rc_attest PRIVATE
    -Wl,--start-group
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_common.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_shielded.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_db.a
    ${BTX_ARCHIVE_DIR}/src/univalue/libunivalue.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_consensus.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_util.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_clientversion.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoinpqc.a
    ${BTX_ARCHIVE_DIR}/src/secp256k1/lib/libsecp256k1.a
    ${BTX_ARCHIVE_DIR}/lib/libbitcoin_crypto.a
    ${BTX_ARCHIVE_DIR}/src/libleveldb.a
    ${BTX_ARCHIVE_DIR}/src/libcrc32c.a
    -Wl,--end-group)
  target_link_options(rc_attest PRIVATE -Wl,--allow-multiple-definition)
endif()  # verify a real block with our ExactReplay, then sign a trusted-mirror attestation
matador_probe(rc_gpu_coupled_probe CUDA)  # GPU coupled-V3 backend byte-exact gate: GPU == CPU == golden a4bb0cc4 + prod-structure cross-check
matador_probe(rc_share_audit_probe THREADS)  # offline ENC_RC share auditor (pool audit-farm sampled-replay tool): JSONL shares -> recomputed digest verdicts; GPU-backed when CUDA
matador_probe(rc_coupled_probe)  # v4.6 ENC_RC coupled V3 golden gate: a4bb0cc4 (medium-V3 dims; production = 1536pg/M=128/P=24)
matador_probe(rc_solve_probe THREADS)  # v4.6 ENC_RC episode SOLVE-LOOP gate: grind/target/winner control flow (toy dims; digest solve contract of SolveMatMulV4RC)
matador_probe(rc_coupled_solve_probe THREADS)  # v4.6 ENC_RC COUPLED-V3 SOLVE-LOOP gate: winner==oracle for the coupled digest (medium-V3 dims)
matador_probe(rc_coupled_bench THREADS)  # MEASURE-ONLY: coupled bank-build vs per-nonce cost (split vs legacy one-shot); the standing CPU/GPU coupled-solver A/B instrument
# red->green probes for the share-path scheduling work (async cpu-confirm,
# share-sink continuation, window->prepare sigma reuse). Each exits 0 on pass.

# local: GPU byte-exact correctness probe for the low-rank correction product (CUDA-core, milestone 2)
# Correctness + speed gate for the in-house int8 GEMM that replaces the cuBLASLt link.
# Links cuBLASLt directly (it benches against it), so it stays a probe -- the miner target
# must NOT grow this dependency back.
if(MATADOR_ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../harness/rc_gemm_i8_probe.cu)
  add_executable(rc_gemm_i8_probe
    ${CMAKE_CURRENT_SOURCE_DIR}/../harness/rc_gemm_i8_probe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../harness/btx_log_stub.cpp)
  target_link_libraries(rc_gemm_i8_probe PRIVATE matador_core CUDA::cublasLt_static)
  # Tuning the tile sweep means recompiling CFG_COUNT CUTLASS kernels; doing that for all five
  # archs makes each iteration minutes long. MATADOR_GEMM_PROBE_ARCH narrows the probe (only
  # the probe -- matador_core keeps the full set) to the card under test.
  set(MATADOR_GEMM_PROBE_ARCH "${MATADOR_CUDA_ARCH}" CACHE STRING "archs for rc_gemm_i8_probe")
  set_target_properties(rc_gemm_i8_probe PROPERTIES CUDA_ARCHITECTURES "${MATADOR_GEMM_PROBE_ARCH}")
  # CUTLASS is header-only: an include path, no library to link and no runtime dependency.
  # MATADOR_CUTLASS_DIR points at a checkout (e.g. -DMATADOR_CUTLASS_DIR=/cutlass).
  if(MATADOR_CUTLASS_DIR AND EXISTS "${MATADOR_CUTLASS_DIR}/include/cutlass/cutlass.h")
    target_include_directories(rc_gemm_i8_probe PRIVATE "${MATADOR_CUTLASS_DIR}/include")
    target_compile_definitions(rc_gemm_i8_probe PRIVATE MATADOR_HAVE_CUTLASS=1)
    message(STATUS "matador: CUTLASS at ${MATADOR_CUTLASS_DIR}")
  else()
    message(STATUS "matador: no CUTLASS (set MATADOR_CUTLASS_DIR) -- probe builds hand kernel only")
  endif()
endif()

if(MATADOR_ENABLE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../harness/lowrank_cuda_probe.cu)
  add_executable(lowrank_cuda_probe
    ${CMAKE_CURRENT_SOURCE_DIR}/../harness/lowrank_cuda_probe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../harness/btx_log_stub.cpp)
  target_link_libraries(lowrank_cuda_probe PRIVATE matador_core)
endif()

# Dependency-free unit tests. Each guards a unit-tested miner gate: VersionGreater,
# InDevFeeWindow (the % charged), DecideUpdate (the v0.6.10 downgrade regression +
# guard ordering), ParsePoolEndpoint/SplitHostPort (pool/proxy connectivity).
#   cmake --build <build> --target <name> && ./<build>/<name>   (exit 0 = pass)
matador_unit_test(version_compare_test)
matador_unit_test(devfee_window_test)
matador_unit_test(update_gate_test)
matador_unit_test(endpoint_parse_test)
matador_unit_test(pool_socket_opts_test)
matador_unit_test(miner_format_test CORE)   # links matador_core for arith_uint256

# Standalone miner binary: links the core; pool/GBT/updater/telemetry orchestration.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../miner/matador-miner.cpp)
  find_package(CURL REQUIRED)
  find_package(Threads REQUIRED)
  # stratum_client.h: ssl:// / tls:// pool connections. On Linux, dynamic (libssl.so.3 ships with
  # every distro, like libcurl already does). macOS ships NO system OpenSSL -- dynamically linking
  # Homebrew's would bake an absolute /opt/homebrew/... dylib path into the "only the BINARY ships"
  # closed-source release, breaking on any Mac without that exact keg installed (the same class of
  # bug as the CUDA green-context fix: a hard runtime dependency on a library not guaranteed
  # present). Static-link it there instead, like CUDA::cudart_static below.
  # STATIC OpenSSL EVERYWHERE (2026-08-05). The build image is Ubuntu 22.04 (OpenSSL 3.0.2), so a
  # dynamic link records libssl.so.3 / libcrypto.so.3 as NEEDED. HiveOS rigs on an Ubuntu 20.04
  # base ship OpenSSL 1.1 and the loader refuses the binary outright:
  #     error while loading shared libraries: libssl.so.3: cannot open shared object file
  # Reported by a user 2026-08-05; the dependency is identical in v0.9.2, so this is not new.
  # Static-linking removes both entries. Safe against the host's libcurl, which carries its own
  # OpenSSL: an executable does not export its symbols unless built -rdynamic (we are not), so our
  # copy cannot interpose on libcurl's.
  set(OPENSSL_USE_STATIC_LIBS TRUE)
  find_package(OpenSSL REQUIRED)
  set(MINER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../miner/matador-miner.cpp)
  add_executable(matador-miner ${MINER_SOURCES})
  target_link_libraries(matador-miner PRIVATE matador_core CURL::libcurl Threads::Threads OpenSSL::SSL OpenSSL::Crypto)
  # Static libstdc++/libgcc for the same portability reason: built on Ubuntu 22.04 the binary
  # otherwise demands GLIBCXX_3.4.30, which Ubuntu 20.04 (max 3.4.28) cannot satisfy. Costs a few
  # MB and removes a whole class of "works on my distro" failures. Not on Apple, where the system
  # libc++ is the only supported one.
  if(NOT APPLE)
    target_link_options(matador-miner PRIVATE -static-libstdc++ -static-libgcc)
  endif()
  if(DEFINED MATADOR_MINER_VERSION)
    target_compile_definitions(matador-miner PRIVATE MATADOR_MINER_VERSION="${MATADOR_MINER_VERSION}")
  endif()
  # The miner shell needs the btx node plumbing (GBT/tx/address/shielded). Point BTX_ARCHIVE_DIR
  # at a built btx tree to link its static archives. Recipe + rationale: build-pathb-miner.sh (#13).
  if(DEFINED BTX_ARCHIVE_DIR)
    # The same btx node archives feed every backend's miner shell (GBT/tx/address/shielded/consensus).
    # matador_core is linked FIRST (above), so its vendored uint256/sha256/Span win the byte-identical
    # dups against btx's pow.cpp by link order on either linker.
    set(BTX_ARCHIVES
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_common.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_shielded.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_db.a
      ${BTX_ARCHIVE_DIR}/src/univalue/libunivalue.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_consensus.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_util.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_clientversion.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoinpqc.a
      ${BTX_ARCHIVE_DIR}/src/secp256k1/lib/libsecp256k1.a
      ${BTX_ARCHIVE_DIR}/lib/libbitcoin_crypto.a
      ${BTX_ARCHIVE_DIR}/src/libleveldb.a
      ${BTX_ARCHIVE_DIR}/src/libcrc32c.a)
    if(MATADOR_ENABLE_CUDA)
      target_link_libraries(matador-miner PRIVATE CUDA::cuda_driver)
      # GNU ld: --start-group resolves the circular btx archive deps; --allow-multiple-definition
      # lets matador_core's vendored dups win.
      target_link_options(matador-miner PRIVATE -Wl,--allow-multiple-definition)
      target_link_libraries(matador-miner PRIVATE -Wl,--start-group ${BTX_ARCHIVES} -Wl,--end-group)
    endif()
  endif()
endif()

if(MATADOR_ENABLE_CUDA)
  target_compile_definitions(matador_core PRIVATE BTX_ENABLE_CUDA_EXPERIMENTAL=1)
  # NOTE: MATADOR_HARDEN (kernel symbol-name obfuscation) went out with the v3
  # kernels -- its mechanism lived in cuda/kernel_names_harden.h, which only
  # matmul_accel.cu / oracle_accel.cu ever included. The rc_* kernels carry no
  # equivalent, so the option was removed rather than left silently doing nothing.
  set_target_properties(matador_core PROPERTIES
    CUDA_ARCHITECTURES "${MATADOR_CUDA_ARCH}"
    CUDA_RUNTIME_LIBRARY Static)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(matador_core PUBLIC CUDA::cudart_static)
  # cuBLASLt provides the INT8 tensor-core GEMMs used by the v4 matmul solvers.
  # STATIC (2026-08-02): mining hosts have no toolkit; dynamic libcublasLt pulls dynamic
  # libcudart and crash-loops on load (the v0.9.0 lesson). PUBLIC so the miner shell and
  # probes inherit the link.
  if(MATADOR_USE_CUBLASLT OR NOT (MATADOR_CUTLASS_DIR AND EXISTS "${MATADOR_CUTLASS_DIR}/include/cutlass/cutlass.h"))
    _matador_cublaslt_lib(_matador_ltlib)
    target_link_libraries(matador_core PUBLIC ${_matador_ltlib})
  endif()
  # NOTE: the [power-prof] board-energy telemetry (MATADOR_ENABLE_POWER_PROF + NVML +
  # core/matmul/power_sampler) was removed with the v3 pipeline it instrumented -- its
  # EnergySpan scopes wrapped the pre-hash scan / prepare / digest-wait stages, none of
  # which exist now. Re-instrumenting the episode loop is open work; a build flag that
  # links NVML and then emits nothing would be worse than not having one.
endif()

