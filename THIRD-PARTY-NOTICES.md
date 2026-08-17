# Third-party notices

AM2 LLC's own code in this repository is MIT licensed (see `LICENSE`). The
components below are third-party and remain under their own terms. Nothing in
`LICENSE` alters them.

## Vendored sources

### btxchain/btx and its Bitcoin Core lineage — MIT

`clean-stack/core/vendor/` is a trimmed vendored subset of
[btxchain/btx](https://github.com/btxchain/btx), which derives from Bitcoin
Core. The upstream copyright notices are retained in the file headers, and the
MIT text those headers reference is in `COPYING`.

Copyright (c) 2009-2010 Satoshi Nakamoto
Copyright (c) 2009-2025 The Bitcoin Core developers
Copyright (c) 2009-2025 Bitcoin Developers

This includes the vendored UniValue tree
(`clean-stack/core/vendor/univalue/`), also MIT.

### tinyformat — Boost Software License 1.0

`clean-stack/core/vendor/tinyformat.h`

Copyright (C) 2011, Chris Foster

Distributed under the Boost Software License, Version 1.0. The full text is at
<https://www.boost.org/LICENSE_1_0.txt>; the notice is retained in the file
header.

## Build-time dependencies (not vendored here)

These are fetched or linked at build time and are not redistributed in this
repository. See `BUILDING.md`.

| Component | License |
|---|---|
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | BSD 3-Clause |
| NVIDIA CUDA Toolkit | NVIDIA CUDA Toolkit EULA |
| OpenSSL | Apache License 2.0 |
| libcurl | curl license (MIT/X derivative) |

Binary releases published by AM2 LLC statically link some of the above; the
corresponding notices travel with those artifacts.

## Reporting a problem

If you believe a notice here is incomplete or incorrect, please open an issue.
