import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import type { UseChatHelpers } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import type { MutableRefObject } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useSendMessage } from "../useSendMessage";

type SendMessageFn = UseChatHelpers<UIMessage>["sendMessage"];
type CreateSessionFn = (options?: {
  expertKickoff?: boolean;
}) => Promise<string | undefined>;

interface HarnessProps {
  sessionId: string | null;
  isSessionScopeReady: boolean;
  organizationId: string | null;
  teamId: string | null;
  sendMessage: SendMessageFn;
  createSession: CreateSessionFn;
  isUserStoppingRef: MutableRefObject<boolean>;
}

function useHarness(props: HarnessProps) {
  return useSendMessage({
    sessionId: props.sessionId,
    isSessionScopeReady: props.isSessionScopeReady,
    sessionOrganizationId: props.organizationId,
    sessionTeamId: props.teamId,
    sendMessage: props.sendMessage,
    createSession: props.createSession,
    isUserStoppingRef: props.isUserStoppingRef,
  });
}

describe("useSendMessage first-send handoff", () => {
  beforeEach(() => {
    useCopilotStreamStore.getState().resetAll();
    window.sessionStorage.clear();
  });

  it.each([
    { label: "personal", organizationId: null, teamId: null },
    { label: "organization", organizationId: "org-a", teamId: null },
    { label: "team", organizationId: "org-a", teamId: "team-a" },
  ])(
    "dispatches the queued $label first send only from the newly mounted session host",
    async ({ organizationId, teamId }) => {
      const newSessionId = `session-${teamId ?? organizationId ?? "personal"}`;
      const sendMessage = vi.fn<SendMessageFn>(async () => undefined);
      const createSession = vi.fn<CreateSessionFn>(async () => {
        useCopilotStreamStore
          .getState()
          .bindPendingFirstSendToSession(newSessionId);
        return newSessionId;
      });
      const isUserStoppingRef = { current: false };
      const props: HarnessProps = {
        sessionId: null,
        isSessionScopeReady: false,
        organizationId,
        teamId,
        sendMessage,
        createSession,
        isUserStoppingRef,
      };
      const oldHost = renderHook(useHarness, { initialProps: props });

      await act(async () => {
        await oldHost.result.current.onSend("first prompt");
      });

      oldHost.rerender({ ...props, sessionId: newSessionId });
      expect(sendMessage).not.toHaveBeenCalled();
      expect(useCopilotStreamStore.getState().pendingFirstSendSessionId).toBe(
        newSessionId,
      );
      oldHost.unmount();

      const newHost = renderHook(useHarness, {
        initialProps: {
          ...props,
          sessionId: newSessionId,
          isSessionScopeReady: false,
        },
      });
      expect(sendMessage).not.toHaveBeenCalled();
      newHost.rerender({
        ...props,
        sessionId: newSessionId,
        isSessionScopeReady: true,
      });

      await waitFor(() => expect(sendMessage).toHaveBeenCalledOnce());
      expect(sendMessage).toHaveBeenCalledWith({
        text: "first prompt",
        files: undefined,
        metadata: undefined,
      });
      expect(useCopilotStreamStore.getState().pendingFirstSend).toBeNull();
    },
  );
});
