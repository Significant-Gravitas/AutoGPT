import { describe, expect, it, vi } from "vitest";

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "http://test.local",
  },
}));

vi.mock("../helpers", () => ({
  getCopilotAuthHeaders: async () => ({ "x-test": "auth" }),
}));

import { createCopilotTransport } from "../copilotStreamTransport";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function makeRefs() {
  return {
    copilotModeRef: { current: undefined },
    copilotModelRef: { current: undefined },
  };
}

function lastMessage(text: string) {
  return [
    {
      id: "ai-sdk-generated-id",
      role: "user" as const,
      parts: [{ type: "text" as const, text }],
    },
  ];
}

describe("copilotStreamTransport.prepareSendMessagesRequest", () => {
  it("attaches a freshly generated UUIDv4 as message_id on every body", async () => {
    const transport = createCopilotTransport({
      sessionId: "sess-1",
      ...makeRefs(),
    });
    // The transport stores the prepared closure on the underlying object;
    // exercise it directly through the public ChatTransport surface.
    const body1 = await (
      transport as unknown as {
        prepareSendMessagesRequest: (args: {
          messages: ReturnType<typeof lastMessage>;
        }) => Promise<{ body: { message_id?: string } }>;
      }
    ).prepareSendMessagesRequest({ messages: lastMessage("hi") });
    expect(body1.body.message_id).toMatch(UUID_RE);
  });

  it(
    "emits a different message_id per call so distinct user clicks dedupe " +
      "as distinct sends server-side",
    async () => {
      const transport = createCopilotTransport({
        sessionId: "sess-1",
        ...makeRefs(),
      });
      const prep = (
        transport as unknown as {
          prepareSendMessagesRequest: (args: {
            messages: ReturnType<typeof lastMessage>;
          }) => Promise<{ body: { message_id?: string } }>;
        }
      ).prepareSendMessagesRequest;

      const a = await prep({ messages: lastMessage("hi") });
      const b = await prep({ messages: lastMessage("hi") });
      expect(a.body.message_id).not.toBe(b.body.message_id);
    },
  );

  it(
    "does NOT pass message_id via AI SDK messageId on sendMessage — " +
      "messageId is replace-mode and would break optimistic render",
    async () => {
      // This is a contract check: the transport reads ``last.id`` (AI SDK's
      // auto-generated id) but must NOT use it as the dedup key, because
      // AI SDK's optimistic-render path treats ``messageId`` on
      // ``sendMessage`` as "edit the existing message with that id".  Since
      // useSendMessage no longer threads a custom messageId, ``last.id`` is
      // an SDK-internal nanoid that's unrelated to our dedup UUID.
      const transport = createCopilotTransport({
        sessionId: "sess-1",
        ...makeRefs(),
      });
      const prep = (
        transport as unknown as {
          prepareSendMessagesRequest: (args: {
            messages: ReturnType<typeof lastMessage>;
          }) => Promise<{ body: { message_id?: string } }>;
        }
      ).prepareSendMessagesRequest;

      const out = await prep({ messages: lastMessage("hi") });
      expect(out.body.message_id).toMatch(UUID_RE);
      // ``last.id`` is "ai-sdk-generated-id" — must NOT be used as message_id.
      expect(out.body.message_id).not.toBe("ai-sdk-generated-id");
    },
  );
});
