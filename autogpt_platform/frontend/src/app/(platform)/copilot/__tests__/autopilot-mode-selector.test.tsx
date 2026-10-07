import {
  getGetV2ListChatTransportsMockHandler,
  getPostV2CreateSessionMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAutopilotModeStore } from "../autopilotModeStore";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

const flags = vi.hoisted(() => ({ autoMode: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.COPILOT_AUTO_MODE ? flags.autoMode : false,
  };
});

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

beforeEach(() => {
  flags.autoMode = true;
});

afterEach(() => {
  resetCopilotChatRegistry();
  useAutopilotModeStore.setState({ choices: {} });
});

describe("AutoPilot mode selector", () => {
  it("is absent with the flag off, and the request carries no mode", async () => {
    flags.autoMode = false;
    const bodies = captureStreamBodies();

    renderHost();
    await typeAndSend("hi");

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(screen.queryByRole("button", { name: /approval mode/i })).toBeNull();
    expect(bodies[0]).not.toHaveProperty("autopilot_mode");
  });

  it("sends the picked mode on the next request", async () => {
    const bodies = captureStreamBodies();

    renderHost();
    await pickMode(/approval mode: auto/i, /ask first/i);

    expect(
      await screen.findByRole("button", { name: /approval mode: ask first/i }),
    ).toBeDefined();
    await typeAndSend("hi");

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toMatchObject({ autopilot_mode: "ask_first" });
  });

  it("leaves the mode unchanged when the Unsupervised confirm is cancelled", async () => {
    const bodies = captureStreamBodies();
    const user = userEvent.setup();

    renderHost();
    await pickMode(/approval mode: auto/i, /unsupervised/i);
    await user.click(await screen.findByRole("button", { name: /cancel/i }));

    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: /run this chat unsupervised/i }),
      ).toBeNull(),
    );
    expect(
      screen.getByRole("button", { name: /approval mode: auto/i }),
    ).toBeDefined();
    await typeAndSend("hi");

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).not.toHaveProperty("autopilot_mode");
  });

  it("switches to Unsupervised only after the confirm", async () => {
    const bodies = captureStreamBodies();
    const user = userEvent.setup();

    renderHost();
    await pickMode(/approval mode: auto/i, /unsupervised/i);
    expect(
      screen.queryByRole("button", { name: /approval mode: unsupervised/i }),
    ).toBeNull();
    await user.click(
      await screen.findByRole("button", { name: /run unsupervised/i }),
    );

    expect(
      await screen.findByRole("button", {
        name: /approval mode: unsupervised/i,
      }),
    ).toBeDefined();
    await typeAndSend("hi");

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toMatchObject({ autopilot_mode: "unsupervised" });
  });

  it("starts from the mode stored on the session", async () => {
    const bodies = captureStreamBodies();

    renderHost({
      sessionOverride: {
        metadata: { dry_run: false, autopilot_mode: "ask_first" },
      },
    });

    expect(
      await screen.findByRole("button", { name: /approval mode: ask first/i }),
    ).toBeDefined();
    await typeAndSend("hi");

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).not.toHaveProperty("autopilot_mode");
  });

  it("carries a new chat's pick into the first request of the session it creates", async () => {
    const bodies = captureStreamBodies();
    server.use(
      getGetV2ListChatTransportsMockHandler({
        transports: [
          {
            auth_provider: "platform",
            credential_id: null,
            label: "AutoGPT Platform",
            available: true,
            default: true,
          },
        ],
      }),
      getPostV2CreateSessionMockHandler({
        id: TEST_SESSION_ID,
        created_at: "2026-05-13T00:00:00Z",
        user_id: "test-user",
      }),
    );

    renderHost({ searchParams: "" });
    await pickMode(/approval mode: auto/i, /ask first/i);
    await typeAndSend("hi");

    await waitFor(() => expect(bodies).toHaveLength(1), { timeout: 5000 });
    expect(bodies[0]).toMatchObject({ autopilot_mode: "ask_first" });
  });
});

describe("autopilotModeStore", () => {
  it("carries a new chat's pick over to the session its first send creates", () => {
    const store = useAutopilotModeStore.getState();
    store.choose(null, "ask_first");
    store.bindNewChatToSession("created-1");

    expect(useAutopilotModeStore.getState().choices).toEqual({
      "created-1": "ask_first",
    });
  });

  it("binds nothing when the new chat had no pick", () => {
    useAutopilotModeStore.getState().bindNewChatToSession("created-1");

    expect(useAutopilotModeStore.getState().choices).toEqual({});
  });
});

function captureStreamBodies() {
  const bodies: Record<string, unknown>[] = [];
  server.use(
    http.post(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
      async ({ request }) => {
        bodies.push((await request.json()) as Record<string, unknown>);
        return streamSseResponse(assistantTextChunks("Done."), {
          abortSignal: request.signal,
        });
      },
    ),
  );
  return bodies;
}

async function pickMode(trigger: RegExp, option: RegExp) {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: trigger }));
  await user.click(await screen.findByRole("menuitemradio", { name: option }));
}
