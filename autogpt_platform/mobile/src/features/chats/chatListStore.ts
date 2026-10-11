import type { BackendClient } from "../../api/backend";
import {
  describeChatError,
  listSessions,
  type SessionSummary,
} from "../../api/chat";
import { Observable } from "../../lib/store";

export interface ChatListState {
  sessions: readonly SessionSummary[];
  total: number;
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  refreshing: boolean;
  loadingMore: boolean;
}

const PAGE = 30;

/** The account's conversations, newest first with pinned ones on top, as the server orders them. */
export class ChatListStore extends Observable<ChatListState> {
  private generation = 0;

  constructor(private readonly api: BackendClient) {
    super({
      sessions: [],
      total: 0,
      status: "idle",
      error: null,
      refreshing: false,
      loadingMore: false,
    });
  }

  async refresh() {
    const generation = ++this.generation;
    const initial = this.getSnapshot().status !== "ready";
    this.set(
      initial
        ? { status: "loading", error: null }
        : { refreshing: true, error: null },
    );
    try {
      const page = await listSessions(this.api, { limit: PAGE, offset: 0 });
      if (generation !== this.generation) return;
      this.set({
        sessions: page.sessions,
        total: page.total,
        status: "ready",
        refreshing: false,
      });
    } catch (error) {
      if (generation !== this.generation) return;
      this.set({
        status: initial ? "error" : "ready",
        error: describeChatError(error),
        refreshing: false,
      });
    }
  }

  async loadMore() {
    const state = this.getSnapshot();
    if (
      state.status !== "ready" ||
      state.loadingMore ||
      state.sessions.length >= state.total
    )
      return;
    const generation = this.generation;
    this.set({ loadingMore: true });
    try {
      const page = await listSessions(this.api, {
        limit: PAGE,
        offset: state.sessions.length,
      });
      if (generation !== this.generation) return;
      const known = new Set(state.sessions.map((s) => s.id));
      this.set({
        sessions: [
          ...state.sessions,
          ...page.sessions.filter((s) => !known.has(s.id)),
        ],
        total: page.total,
        loadingMore: false,
      });
    } catch (error) {
      if (generation !== this.generation) return;
      this.set({ loadingMore: false, error: describeChatError(error) });
    }
  }
}
