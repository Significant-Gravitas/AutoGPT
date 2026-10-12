import type { UIMessage } from "ai";
import { vi } from "vitest";

import { getCopilotAuthHeaders } from "../../../helpers";
import { convertChatSessionMessagesToUiMessages } from "../../../helpers/convertChatSessionToUiMessages";
import { createAssistantRunJoiner, createTailRenderer } from "../../renderTail";
import {
  TurnRuntime,
  type RuntimeDeps,
  type SessionView,
} from "../../turnRuntime";
import type { FaultServer } from "./faultServer";

const SESSION = "session-1";

/** What the user sees: the error box, the inline notice, the status, the toast count, and the chat. */
export interface Screen {
  error: string | null;
  notice: string | null;
  phase: string;
  toasts: number;
  chat: string[];
}

/**
 * One browser tab on the chat, bound as `useCopilotRuntimeStream` binds the
 * runtime: the runtime's session reads are the page's refetches, and the
 * page observes each view it gets on the next macrotask, as React Query
 * notifies. The chat is the view's history above the runtime's tail, both
 * through the renderers the page uses.
 */
export class Tab {
  readonly runtime: TurnRuntime;
  private toast = vi.fn<RuntimeDeps["toast"]>();
  private lastView: SessionView | null = null;
  private renderTail = createTailRenderer(SESSION);
  private joinRuns = createAssistantRunJoiner();

  constructor(
    private readonly server: FaultServer,
    private readonly viewLatencyMs = 0,
  ) {
    this.runtime = new TurnRuntime(SESSION, {
      baseUrl: () => "http://backend.test",
      headers: getCopilotAuthHeaders,
      fetch: server.fetch,
      fetchSession: () => this.refetch(),
      toast: this.toast,
      report: vi.fn(),
    });
    this.runtime.bind({}, () => this.refetch());
    this.lastView = server.view();
    this.runtime.observe(this.lastView);
  }

  send(text: string) {
    void this.runtime.send({ text }, undefined);
  }

  screen(): Screen {
    const snapshot = this.runtime.getSnapshot();
    const cut = snapshot.ownedFrom;
    const history = (
      (this.lastView?.messages ?? []) as { sequence?: number }[]
    ).filter((row) => cut === null || (row.sequence ?? 0) < cut);
    const messages = this.joinRuns([
      ...convertChatSessionMessagesToUiMessages(SESSION, history, {
        isComplete: !this.lastView?.active_stream,
      }).messages,
      ...this.renderTail(snapshot).messages,
    ]);
    return {
      error: snapshot.error?.message ?? null,
      notice: snapshot.notice,
      phase: snapshot.phase,
      toasts: this.toast.mock.calls.length,
      chat: messages.map(textOf),
    };
  }

  dispose() {
    this.runtime.dispose();
  }

  private async refetch(): Promise<SessionView | null> {
    if (this.viewLatencyMs)
      await new Promise((resolve) => setTimeout(resolve, this.viewLatencyMs));
    let view: SessionView;
    try {
      view = this.server.view();
    } catch {
      return null;
    }
    this.lastView = view;
    setTimeout(() => this.runtime.observe(view), 0);
    return view;
  }
}

function textOf(message: UIMessage) {
  const text = message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
  return `${message.role}: ${text}`;
}
