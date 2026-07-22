# E2B vs Daytona — Pricing Comparison for AutoGPT Interactive Desktop Sandboxes

Research date: 2026-07-22. All rates USD. "PUBLISHED" = read directly from the provider's own page (URL given). Third-party cross-checks: [Northflank AI sandbox pricing (2026)](https://northflank.com/blog/ai-sandbox-pricing), [ZenML E2B vs Daytona](https://www.zenml.io/blog/e2b-vs-daytona).

**Headline finding: the metered Linux compute rates are identical on both platforms** — $0.000014/vCPU/s and $0.0000045/GiB-RAM/s ($0.0504/vCPU-h, $0.0162/GiB-h). The cost difference is entirely in fixed fees, disk/persistence billing, and plan gates. (Daytona's Linux rates were extracted from the raw HTML of daytona.io/pricing — the page is JS-rendered and shows both per-second and per-hour views, which are mutually consistent: 0.000014×3600=0.0504 ✓.)

---

## 1. Rate table

| Rate | E2B | Daytona | Status |
|---|---|---|---|
| vCPU | **$0.000014/s** = $0.0504/h | **$0.00001400/s** = $0.0504/h | PUBLISHED — [e2b.dev/pricing](https://e2b.dev/pricing); [daytona.io/pricing](https://www.daytona.io/pricing). Cross-checked: Northflank, ZenML |
| RAM (per GiB) | **$0.0000045/s** = $0.0162/h | **$0.00000450/s** = $0.0162/h | PUBLISHED — same URLs, same cross-checks |
| Disk (per GiB) | **No metered rate.** 10 GiB included (Hobby) / 20 GiB (Pro); "20+" via support. Disk is an allocation *cap*, not a metered line item | **$0.00000003/s** = $0.000108/GiB/h, "*5 GB Free then price for each GB/s after" | PUBLISHED — [e2b.dev/pricing](https://e2b.dev/pricing) + [e2b.dev/docs/billing](https://e2b.dev/docs/billing); [daytona.io/pricing](https://www.daytona.io/pricing). Cross-checked: ZenML, Northflank |
| Windows OS surcharge | n/a (not offered) | **$0.0858/vCPU/h** ($0.0000238/vCPU/s), listed under "OS" | PUBLISHED — [daytona.io/pricing](https://www.daytona.io/pricing). Additive-vs-all-in NOT stated (§5) |
| GPU (context only) | not published | H200 $4.54/h · H100 $3.95/h · RTX PRO 6000 $3.03/h · RTX 5090 $1.29/h · RTX 4090 $0.99/h | PUBLISHED — [daytona.io/pricing](https://www.daytona.io/pricing) |
| Paused/state billing | Paused = **$0** — billing FAQ: "You only pay while a sandbox is actively running"; paused sandboxes kept **indefinitely** | Started/Creating/Starting/Stopping/Pausing = full vCPU+RAM+disk; **Stopped = disk only**; **Archived = $0**; Deleted = $0 (snapshots still billed for storage) | PUBLISHED — [e2b.dev/docs/billing](https://e2b.dev/docs/billing), [e2b.dev/docs/sandbox/persistence](https://e2b.dev/docs/sandbox/persistence); [daytona.io/docs/en/billing.md](https://www.daytona.io/docs/en/billing.md) |
| Volumes | **Private beta, no published price** | **Free** — "included at no additional cost", up to **100 volumes/org**, volume data does **not** count against storage quota | PUBLISHED — [daytona.io/docs/en/volumes.md](https://www.daytona.io/docs/en/volumes.md) |
| Egress / network | Not published | Not published — [network-limits.md](https://www.daytona.io/docs/en/network-limits.md) is firewall/allow-list config only | Neither provider publishes egress pricing |
| Billing increment | Per-second ("pay per second… while your sandbox is running") | Per-second rates displayed; usage metered in CPU-seconds / GB-seconds ([billing.md](https://www.daytona.io/docs/en/billing.md)) | PUBLISHED (E2B); implied-by-metering (Daytona) |
| Sign-up credits | **$100 one-time** (Hobby) | **$200** free compute; startup program **up to $50k** ("$10K straight away") | PUBLISHED — [e2b.dev/pricing](https://e2b.dev/pricing); [daytona.io/pricing](https://www.daytona.io/pricing), [daytona.io/startups](https://www.daytona.io/startups) |
| Fixed fees | Hobby $0 · **Pro $150/mo (includes NO usage credits — usage billed on top)** · Enterprise custom | **None** — pure PAYG; tier upgrades are wallet top-ups ($25/$500/$2,000) that remain **spendable credits**, not fees | PUBLISHED — [e2b.dev/docs/billing](https://e2b.dev/docs/billing); [daytona.io/docs/en/limits.md](https://www.daytona.io/docs/en/limits.md) |

Reference sandbox for all scenarios: **2 vCPU + 4 GiB RAM + 10 GiB disk**.
Common compute rate (both providers): 2 × $0.0504 + 4 × $0.0162 = $0.1008 + $0.0648 = **$0.1656/h**.
Daytona disk: assuming "first 5 GB free" applies per sandbox (scope unpublished, §5), billable disk = 5 GiB × $0.000108/h = **$0.00054/h** (worst case, all 10 GiB billed: $0.00108/h — never moves a rounded result by more than $0.40/mo).

---

## 2. Scenarios

### S1 — One desktop, 1 hour running

| | E2B | Daytona |
|---|---|---|
| Compute | 1 × $0.1656 = $0.1656 | 1 × $0.1656 = $0.1656 |
| Disk | $0 (10 GiB within included allocation) | 5 GiB × $0.000108 = $0.00054 |
| **Total** | **$0.17** | **$0.17** |
| Plan gate | Hobby OK (≤1 h session; disk exactly at Hobby's 10 GB cap) | Tier 1 OK; fits default per-sandbox max (4 vCPU / 8 GB / 10 GB — disk exactly at cap) |

### S2 — Same desktop, 8 h/day × 22 workdays (176 running hours; 30-day month = 720 h)

| | E2B | Daytona |
|---|---|---|
| Compute | 176 × $0.1656 = $29.1456 | 176 × $0.1656 = $29.1456 |
| Disk (running) | $0 | 176 × $0.00054 = $0.0950 |
| Disk (idle: 720−176 = 544 h) | $0 (paused = free) | 544 × $0.00054 = $0.2938 (stopped) |
| Fixed fee | **$150 (Pro required — 8 h continuous > Hobby's 1 h max runtime)** | $0 |
| **Total** | **$179.15** | **$29.53** *(worst case all-10-GiB-billed: $29.92; archived-when-idle: $29.24)* |

E2B caveat: on Hobby you'd have to pause/resume every hour (1 h max runtime) — usage alone would be $29.15/mo against the one-time $100 credits, but that's not viable UX for an interactive desktop; treat Pro as required.

### S3 — Idle-but-persistent, 30 days (720 h)

| Mode | Math | Monthly cost |
|---|---|---|
| E2B paused | published: $0 while not running, kept indefinitely | **$0.00** |
| Daytona stopped (container) | 5 GiB billable × $0.000108 × 720 | **$0.39** *(all 10 GiB: $0.78)* |
| Daytona archived (container → object storage) | not billed | **$0.00** (slower resume; container sandboxes only) |

### S4 — 100 users × persistent 5 GiB workspace volume

| | E2B | Daytona |
|---|---|---|
| Native volumes | **Cannot price — private beta, no published rate** | 100 × 5 GiB = **$0.00** (volumes free; exactly at the 100-volume/org cap — Daytona docs' recommended pattern is one shared volume with per-user `subpath` mounts, which also stays under the cap) |
| Workaround | ASSUMED: 100 *paused* sandboxes as per-user persistence → $0 published (paused free, kept indefinitely), each ≤10/20 GiB disk cap | — |
| **Total** | **$0 (assumed, via paused sandboxes) / unpriceable via volumes** | **$0.00 (PUBLISHED)** |

### S5 — Burst: 50 concurrent desktops × 2 h (aggregate: 100 vCPU, 200 GiB RAM, 500 GiB disk)

| | E2B | Daytona |
|---|---|---|
| Compute | 50 × 2 × $0.1656 = $16.56 | 50 × 2 × $0.1656 = $16.56 |
| Disk | $0 | 50 × 5 GiB × $0.000108 × 2 = $0.054 |
| **Usage total** | **$16.56** | **$16.61** |
| Gate | **Hobby caps at 20 concurrent → Pro ($150/mo) required**; Pro allows 100 (purchasable to 1,100). Creation rate 1/s Hobby, 5/s Pro | No concurrency count cap — tier resource pool. Tier 2 (card + $25 top-up: 100 vCPU/200 GiB/300 GiB) is exactly at vCPU+RAM caps and **over on disk (500 > 300 GiB) → Tier 3 needed ($500 cumulative top-up — spendable credits, not a fee)**; or shrink disk to ≤6 GiB/desktop to squeeze into Tier 2 with zero headroom. Creation ≥300/min all tiers |
| **Cash out** | **$166.56 first month** ($16.56 if already on Pro) | **$16.61 of credit** (but ≥$500 must sit in the wallet for Tier 3) |

---

## 3. Plan & quota gates

**E2B** ([pricing](https://e2b.dev/pricing), [billing docs](https://e2b.dev/docs/billing) — PUBLISHED, mutually consistent):

- Hobby (free, $100 one-time credits): **1 h max continuous runtime**, **20 concurrent**, 10 GB disk, 8 vCPU / 8 GB RAM max per sandbox, 1 create/s.
- Pro ($150/mo + usage, no included credits): **24 h max runtime**, **100 concurrent** (extra purchasable to 1,100), 20 GB disk ("+" via support), 5 creates/s.
- The desktop feature effectively **requires Pro** (any session >1 h, or >20 concurrent users).

**Daytona** ([limits.md](https://www.daytona.io/docs/en/limits.md) — PUBLISHED):

- Tiers are org-wide resource pools, not sandbox counts: Tier 1 (email verified): 10 vCPU / 10–20 GiB RAM (docs self-conflict, §5) / 30 GiB. Tier 2 (card + $25 top-up): 100 / 200 GiB / 300 GiB. Tier 3 ($500 top-up): 250 / 500 GiB / 2,000 GiB. Tier 4 ($2,000 top-up per 30 days): 500 / 1,000 GiB / 5,000 GiB.
- Per-sandbox default max **4 vCPU / 8 GB RAM / 10 GB disk** (the 2/4/10 desktop fits, disk exactly at cap).
- No published max session length; auto-stop/auto-archive/auto-delete intervals configurable per sandbox.
- Rate limits generous (≥300 sandbox creations/min at Tier 1).
- Stopped **container** sandboxes keep occupying org disk quota until archived (quota ≠ billing; tracked separately).

---

## 4. Which is cheaper when

- **Metered compute is a literal wash** — both charge $0.0504/vCPU-h + $0.0162/GiB-h per second. No usage-rate arbitrage for running desktops.
- **Daytona is cheaper in every modeled scenario**, entirely because of E2B's $150/mo Pro fee, which the desktop feature can't avoid (1 h Hobby session cap, 20-concurrent cap). At AutoGPT-shaped usage (~$29/mo metered), E2B's fixed fee is ~5× the usage itself.
- **E2B wins only on paused persistence purity** ($0 forever, published) vs Daytona stopped ($0.39–0.78/mo per desktop) — and Daytona *archived* is also $0, so even this edge disappears if slower resume is tolerable.
- **Persistence/workspaces**: Daytona strictly ahead today — free volumes (100/org, documented subpath multi-tenancy) vs E2B volumes in private beta with no price.
- **Pilot runway**: Daytona $200 free (~1,200 desktop-hours at $0.1656/h) vs E2B $100 (~600 h) — plus Daytona's startup program (up to $50k, $10k upfront) could cover a full beta. Daytona's tier top-ups stay spendable; E2B's $150/mo does not.
- **When E2B could win**: a hard 24 h session ceiling as a built-in safety rail; >500 vCPU without enterprise negotiation (Daytona Tier 4 caps at 500 vCPU with $2k-per-30-days top-ups); negotiated Enterprise pricing. Also note Daytona bills full rate during Creating/Starting/Stopping/Pausing transitions; E2B's "only while running" wording is slightly more favorable (delta is seconds).

---

## 5. Numbers that could NOT be confirmed

1. **Daytona "first 5 GB free" disk — scope unstated** (per sandbox vs per org). Pricing-page footnote only. Both interpretations computed; max impact ~$0.39/sandbox/mo.
2. **Daytona Windows rate semantics** — $0.0858/vCPU/h listed under "OS"; additive to or replacing the $0.0504 Linux vCPU rate is unpublished. (Moot for Linux desktops.)
3. **E2B storage beyond included 10/20 GiB** — no per-GiB overage rate exists anywhere; disk is a plan allocation ("20+ GB — contact support"), so >20 GiB desktops are unpriceable without sales contact.
4. **Egress/network transfer pricing — both providers**: nothing published (Daytona network-limits.md is firewall config only).
5. **E2B paused-sandbox storage**: official billing FAQ says $0 ("only pay while actively running"; kept indefinitely), but Northflank's third-party comparison claims "storage costs accruing while paused." Official doc treated as authoritative; worth confirming with E2B before betting persistence architecture on free-forever pause.
6. **E2B volumes pricing** — private beta, no numbers.
7. **Daytona Tier 1 RAM** — limits.md self-conflicts: Tiers table says 10 GiB, Limits table says 20 GiB.
8. **Daytona formal minimum billing increment & max session length** — per-second rates and CPU-seconds metering are published, but no explicit "minimum increment" or session ceiling statement.
9. **E2B Hobby post-credit behavior** — whether you can pay-as-you-go on Hobby after the $100 one-time credits without upgrading to Pro is not documented.
