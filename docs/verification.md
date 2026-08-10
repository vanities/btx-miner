# Verifying ENC_RC work with matador

BTX v4 (ENC_RC) proof of work is an *episode replay*, not a hash. Checking it means
re-running the episode and comparing the resulting digest. `matador-miner --verify` does
exactly that, using the same episode backend the solver itself uses, so a verifier cannot
drift away from the code that produced the work.

It is a one shot command. It needs no pool, no payout address and no config file.

## Requirements

* A CUDA GPU (Ampere or newer) with driver r550 or newer. Without a GPU the command falls
  back to the portable CPU oracle, which is correct but far slower.
* Any matador build whose `--help` lists `--verify`.

## Node operators: is this block header valid?

```bash
matador-miner --verify \
  --header     <raw 182 byte block header, hex> \
  --height     <block height> \
  --parent-mtp <median time past of the PARENT block>
```

Getting the inputs from a node, for a block at `$HASH`:

```bash
HDR=$(btx-cli getblockheader "$HASH" false)                 # raw header hex
PREV=$(btx-cli getblockheader "$HASH" | jq -r .previousblockhash)
MTP=$(btx-cli getblockheader "$PREV" | jq -r .mediantime)   # PARENT median time past
HEIGHT=$(btx-cli getblockheader "$HASH" | jq -r .height)

matador-miner --verify --header "$HDR" --height "$HEIGHT" --parent-mtp "$MTP"
```

`getblockheader <hash> false` returns the raw header even for a block the node has not
connected yet, so this works on a tip you are still deciding about.

Output:

```
[verify] height=185000 matmul_dim=4096
[verify] sigma           = 3f223a511f7b8907b81c377301106bdc...
[verify] replayed digest = 00009f2b6a4bf5f206e4a047d0eab260...
[verify] header digest   = 00009f2b6a4bf5f206e4a047d0eab260...
[verify] digest_matches  = YES
[verify] beats_block_target = YES
[verify] VERDICT: VALID
```

`digest_matches` is the important line. It means the header's committed digest is what the
episode actually produces for that header. `beats_block_target` then says the work clears
the consensus target from the header's own `nBits`.

## Pool operators: did this share do real work?

Add `--share-target`. Assemble the header from the job you issued plus the `nonce64` and
`ntime` the miner submitted, then:

```bash
matador-miner --verify \
  --header       <header built from your job + the submitted nonce64/ntime> \
  --height       <job height> \
  --parent-mtp   <parent median time past for that job> \
  --share-target <64 hex characters, the target you issued>
```

Adds one line:

```
[verify] beats_share_target = YES
[verify] VERDICT: VALID
```

Two independent checks, and you want both:

* `digest_matches = YES` proves the miner ran the episode and did not fabricate a digest.
* `beats_share_target = YES` proves the result clears the difficulty you issued.

A share that satisfies your target but whose digest does not match the replay is not real
work. Grading on the submitted digest alone cannot detect that.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | valid |
| 1 | invalid (digest mismatch, or target not met) |
| 2 | bad arguments (malformed hex, missing flag) |
| 3 | could not verify (bad `nBits`, seed derivation failed, no backend) |

Suitable for scripting: `if matador-miner --verify ...; then ...`

## Notes

* `--parent-mtp` is the **parent** block's median time past, not the block's own timestamp
  and not the parent's `time`. Passing the wrong value changes the seed derivation and the
  digest will not match, which reads as INVALID.
* `--share-target` must be exactly 64 hex characters.
* Verification cost is one episode, the same unit of work a miner performs per nonce
  attempt. Budget for that when grading shares at volume.
