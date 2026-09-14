# Design review: memory-backed Expert learning

Fable 5.1, through Claude Code, reviewed the proposal twice on September 14, 2026. This is a review of a design, not evidence that an implementation exists or has passed its future release tests.

The second review found the design coherent enough for a draft design PR, conditional on resolving the remaining wording gaps. Those findings are incorporated into the [proposal](expert-learning.md).

## Changes made after review

| Finding                                                                | Resolution in the proposal                                                                                                                |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| “Verified” was circular when derived from an agent's own success claim | Separate observed source success, an accepted artifact, concrete checks, and later per-version reuse. A load is never counted as success. |
| Users could encounter changed behavior without seeing its origin       | Add an in-conversation skill-loaded chip naming the version and origin, linked to existing detail/history surfaces.                       |
| Human edits and automated updates had an unclear boundary              | Use a visible per-skill automatic-improvement policy, preserve immediate human edits, and coalesce exceptional proposals.                 |
| Undo could be evaded by a trivially new evidence fingerprint           | Suppress the normalized behavior within its scope; equivalent proposals need explicit review even with new evidence.                      |
| Revocation omitted human-edited descendants                            | Descendants inherit source dependencies; invalidation affects them, with independently supported restoration or re-authoring.             |
| Evidence availability confused viewer access with owner access         | Keep owner eligibility separate from what a particular viewer may see; do not reveal inaccessible source metadata.                        |
| Static approval fields could not satisfy the apply-time race cases     | Require live source revalidation and a fresh eligibility precondition alongside expected skill version.                                   |
| Non-software work could never meet the proposed evidence threshold     | An accepted artifact can support its production method, without verifying the artifact's claims about the world.                          |
| Secret handling depended too heavily on reviewer-model judgment        | Require content checks, a truthful blocked disposition, and an audited non-secret false-positive path.                                    |
| Pause, correction, outcome reporting, and forgetting were ambiguous    | Distinguish learning from use, immediate edit from outcome evidence, exclusion from revocation, and restoration from erasure.             |
| The first release had too many secondary UX features                   | Defer a separate digest, broad sharing, merges/renames, age-based archiving, per-section provenance, and selective forget UI.             |

## Recommendations deliberately adapted

Human edits do not freeze a skill forever: the user can explicitly allow future automatic improvements. Ordinary source exclusion does not erase existing knowledge. A pinned version does not justify continuing after explicit stop-use or approval/access revocation. These choices are documented as product rules, with acceptance scenarios.

The review remains advisory. Implementation must still demonstrate the complete evidence-to-skill-to-reuse-to-restore path, including failure, race, scope, and mobile-interaction cases. No runtime or configuration change is part of this design PR.
