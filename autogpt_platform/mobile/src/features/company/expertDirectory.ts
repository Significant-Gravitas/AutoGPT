import type { BackendClient } from "../../api/backend";
import { errorMessage } from "../../api/errors";
import {
  listExperts,
  listExpertIdentities,
  type ExpertIdentity,
  type ExpertProfile,
} from "../../api/experts";
import { Observable } from "../../lib/store";

export interface ExpertDirectoryState {
  /** Includes let-go experts, so their old conversations still show who they were. */
  identities: ReadonlyMap<string, ExpertIdentity>;
  roster: readonly ExpertProfile[];
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
}

/** The signed-in account's AI company: Otto is implicit, experts come from the server. */
export class ExpertDirectory extends Observable<ExpertDirectoryState> {
  private inflight: Promise<void> | null = null;

  constructor(private readonly api: BackendClient) {
    super({ identities: new Map(), roster: [], status: "idle", error: null });
  }

  load(): Promise<void> {
    if (!this.inflight) {
      this.inflight = this.fetch().finally(() => {
        this.inflight = null;
      });
    }
    return this.inflight;
  }

  /** Fetches once; later callers reuse what is loaded. */
  ensureLoaded() {
    if (this.getSnapshot().status === "idle") void this.load();
  }

  private async fetch() {
    const current = this.getSnapshot();
    if (current.status === "idle" || current.status === "error") {
      this.set({ status: "loading", error: null });
    }
    const [identities, roster] = await Promise.allSettled([
      listExpertIdentities(this.api),
      listExperts(this.api),
    ]);
    if (identities.status === "rejected" && roster.status === "rejected") {
      // A failed refresh keeps whatever is already shown.
      const shown = this.getSnapshot();
      this.set({
        status:
          shown.roster.length > 0 || shown.identities.size > 0
            ? "ready"
            : "error",
        error: errorMessage(identities.reason, "Couldn't load your experts."),
      });
      return;
    }
    const rosterValue =
      roster.status === "fulfilled" ? roster.value : this.getSnapshot().roster;
    const identityList =
      identities.status === "fulfilled" ? identities.value : rosterValue;
    this.set({
      identities: new Map(
        identityList.map((identity) => [identity.id, identity]),
      ),
      roster: rosterValue,
      status: "ready",
      error:
        roster.status === "rejected"
          ? errorMessage(roster.reason, "Couldn't load your experts.")
          : null,
    });
  }
}
