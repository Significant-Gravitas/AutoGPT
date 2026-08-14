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
import { buildKickoffMessage } from "../expertKickoff";

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
  it("reaches fetch when crypto.randomUUID is unavailable on a LAN HTTP origin", async () => {
    const originalCrypto = globalThis.crypto;
    const fetchReached = new Error("fetch reached");
    const fetchMock = vi.fn().mockRejectedValue(fetchReached);
    vi.stubGlobal("crypto", {
      getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
    });
    vi.stubGlobal("fetch", fetchMock);

    try {
      const transport = createCopilotTransport({
        sessionId: "sess-lan-http",
        ...makeRefs(),
      });

      await expect(
        transport.sendMessages({
          trigger: "submit-message",
          chatId: "sess-lan-http",
          messageId: undefined,
          messages: lastMessage("hello from a phone"),
          abortSignal: undefined,
        }),
      ).rejects.toBe(fetchReached);
      expect(fetchMock).toHaveBeenCalledOnce();
      expect(fetchMock).toHaveBeenCalledWith(
        "http://test.local/api/chat/sessions/sess-lan-http/stream",
        expect.objectContaining({ method: "POST" }),
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

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

describe("copilotStreamTransport — expert kickoff dedup ids", () => {
  const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";

  function prep(transport: ReturnType<typeof createCopilotTransport>) {
    return (
      transport as unknown as {
        prepareSendMessagesRequest: (args: {
          messages: Array<{
            id: string;
            role: "user" | "assistant";
            parts: Array<{ type: "text"; text: string }>;
          }>;
        }) => Promise<{ body: { message_id?: string } }>;
      }
    ).prepareSendMessagesRequest;
  }

  it("derives the same attempt-0 id for the first kickoff in any tab", async () => {
    const transport = createCopilotTransport({
      sessionId: "sess-1",
      ...makeRefs(),
    });
    const kickoff = buildKickoffMessage(EXPERT_ID);

    const tabA = await prep(transport)({ messages: lastMessage(kickoff) });
    const tabB = await prep(transport)({ messages: lastMessage(kickoff) });

    // Two tabs racing the same first kickoff collide on the PK server-side,
    // so the turn (and its workflow side effects) fires exactly once.
    expect(tabA.body.message_id).toBe(`expert-kickoff-${EXPERT_ID}-0`);
    expect(tabB.body.message_id).toBe(tabA.body.message_id);
  });

  it("advances the attempt on retry so a failed kickoff is not dead-ended by dedup", async () => {
    const transport = createCopilotTransport({
      sessionId: "sess-1",
      ...makeRefs(),
    });
    const kickoff = buildKickoffMessage(EXPERT_ID);

    const retry = await prep(transport)({
      messages: [
        {
          id: "m1",
          role: "user" as const,
          parts: [{ type: "text" as const, text: kickoff }],
        },
        {
          id: "m2",
          role: "assistant" as const,
          parts: [{ type: "text" as const, text: "stream failed" }],
        },
        {
          id: "m3",
          role: "user" as const,
          parts: [{ type: "text" as const, text: kickoff }],
        },
      ],
    });

    expect(retry.body.message_id).toBe(`expert-kickoff-${EXPERT_ID}-1`);
  });

  it("keeps random UUIDs for ordinary messages", async () => {
    const transport = createCopilotTransport({
      sessionId: "sess-1",
      ...makeRefs(),
    });

    const out = await prep(transport)({
      messages: lastMessage("You were just hired."),
    });

    expect(out.body.message_id).toMatch(UUID_RE);
  });
});
