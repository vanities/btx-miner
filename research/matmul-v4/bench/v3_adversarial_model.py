#!/usr/bin/env python3
# V4.5 V3 adversarial economics model (Workstream F/J/K). Inputs are MEASURED primitives;
# usage: v3_adversarial_model.py <comp5090_ms> <regen_page_ms> <b200_gemm_ms> [bank_gib] [dph5090] [dphB200]
import sys
comp5090   = float(sys.argv[1]) if len(sys.argv)>1 else 51.2
regen_page = float(sys.argv[2]) if len(sys.argv)>2 else 1.47
b200_gemm  = float(sys.argv[3]) if len(sys.argv)>3 else None   # per-page GEMM ms on B200 (measured)
BANK=float(sys.argv[4]) if len(sys.argv)>4 else 51.0
dph5090=float(sys.argv[5]) if len(sys.argv)>5 else 0.31
dphB200=float(sys.argv[6]) if len(sys.argv)>6 else 6.25
PAGES=1536
cache=min(1.0,32.0/BANK); overflow=PAGES*(1-cache); regen=overflow*regen_page
rent=dphB200/dph5090
compB200 = (b200_gemm*PAGES) if b200_gemm else comp5090/3.0   # measured, else assume 3x
b200_ns=1000.0/compB200
print(f"5090 caches {cache*100:.0f}% of {BANK} GiB packed; overflow {overflow:.0f} pages, "
      f"regen {regen:.0f} ms/nonce unbatched | rent B200/5090={rent:.1f}x")
print(f"B200 compute {compB200:.1f} ms/nonce -> {b200_ns:.1f} nonce/s (resident); "
      f"B200 is {comp5090/compB200:.1f}x a 5090 on compute\n")
print(f"{'Q':>6} {'5090 n/s':>9} {'rateB200/5090':>13} {'per-$ 5090/B200':>16} {'WINNER/$':>9}")
for Q in (1,8,32,128,1024):
    ns=1000.0/(comp5090+regen/Q); rr=b200_ns/ns; pd=(ns/dph5090)/(b200_ns/dphB200)
    print(f"{Q:>6} {ns:>9.2f} {rr:>12.1f}x {pd:>15.1f}x {'B200' if rr>rent else '5090':>9}")
