# Upstream (teammate `fwq`) sync log

Remote: `origin` = https://github.com/InkyUmbrella/2026Ncccu_BiDataCompetition.git
Our branch: `heyuhang/w1mer-opt` (branched from `origin/fwq` @ 9141f72).
Bridge: local repo pushes to `server` = ssh://heyuhang@100.68.97.6/home/heyuhang/work/ncccu-repo.git;
the GPU server worktree `/home/heyuhang/work/ncccu` pulls from it. GitHub push is impossible
from this account (no credentials), so the bare repo + bundles are the sharing channel.

## Sync policy
1. Before every batch: `git fetch origin` and `git log --oneline HEAD..origin/fwq`.
2. If `origin/fwq` moved: merge it into our branch (`git merge origin/fwq`) and resolve in favour of
   keeping both sides' semantics (never drop teammate features).
3. Teammate's working tree is readable: `/home/zhenxiaojun/projects/2026Ncccu_BiDataCompetition`
   (read-only for us). We may fetch its refs to see unpushed commits, but never write there.
4. Our commits are pushed to the `server` bare repo each batch; a `git bundle` is left in
   `/var/tmp/ncccu-heyuhang/` so teammates can pull without GitHub access.

## 2026-10-22 status
- `origin/fwq` = 9141f72 ("年龄参差"), unchanged since 2026-10-21; teammate working tree clean.
- Our branch adds (all default-off or additive): EMA, stochastic depth, mixup/cutmix, label
  smoothing, decode policy (full|draft) + parity/latency harness, run harness, 4 experiment configs,
  60 CPU tests, data audit, review R_01/R_02, SSL plan.
- Teammate code that exists only in `outputs/**/code` snapshots (not committed): `engine/age_validation.py`
  and the `age_residual_enabled` flag. We intentionally do not import those yet; if we adopt
  age-residual validation we will port them with attribution in the merge commit.
- Baseline reference: crop384 from-scratch ConvNeXt-T (SD 0.1 default), 60 ep, pos_weight 5,
  valid AUROC 0.9233 (teammate, artifact not readable by us). Positive sampling 1/7 + 20 ep
  fine-tune reached 0.9296. Our four arms reproduce this under committed artifacts.
