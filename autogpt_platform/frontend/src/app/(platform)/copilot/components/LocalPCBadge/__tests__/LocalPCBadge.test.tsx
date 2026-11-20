import { getGetExperimentalGetSessionExecutorQueryKey } from "@/app/api/__generated__/endpoints/copilot/copilot";
import {
  getGetExperimentalGetSessionExecutorMockHandler200,
  getGetExperimentalGetSessionExecutorMockHandler401,
} from "@/app/api/__generated__/endpoints/copilot/copilot.msw";
import { getGetFeatureFlagsMockHandler } from "@/app/api/__generated__/endpoints/feature-flags/feature-flags.msw";
import { server } from "@/mocks/mock-server";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import {
  QueryClient,
  QueryClientProvider,
  focusManager,
} from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { LocalPCBadge } from "../LocalPCBadge";

const EXECUTOR_URL =
  "http://localhost:3000/api/proxy/api/copilot/sessions/:sessionId/executor";
const SESSION_ID = "11111111-1111-4111-8111-111111111111";

function serveRolloutFlag(flags: Partial<Record<Flag, unknown>>) {
  server.use(getGetFeatureFlagsMockHandler({ flags, configured: true }));
}

describe("LocalPCBadge", () => {
  beforeEach(() => {
    serveRolloutFlag({ [Flag.LOCAL_PC_EXECUTOR]: true });
  });

  test("explains how to reconnect the established Local PC session", async () => {
    const user = userEvent.setup();
    server.use(
      getGetExperimentalGetSessionExecutorMockHandler200({
        kind: "none",
        computer_use_consent: "pending",
      }),
    );

    render(
      <LocalPCBadge
        sessionID={SESSION_ID}
        machineID="machine-123456789"
        allowedRoot={"C:\\Users\\Ada\\Projects"}
      />,
    );

    const trigger = await screen.findByRole("button", {
      name: /local pc disconnected.*open local pc details/i,
    });
    await user.click(trigger);

    expect(
      await screen.findByText(/restart autogpt-shim.*to reconnect/i),
    ).toBeDefined();
    expect(screen.getByText(/machine machine-1234/i)).toBeDefined();
    expect(screen.getByText("C:\\Users\\Ada\\Projects")).toBeDefined();
  });

  test("opens details for the connected machine", async () => {
    const user = userEvent.setup();
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          platform: "darwin",
          arch: "arm64",
          allowed_root: "/Users/test/autogpt-workspace",
          machine_id: "abcdef1234567890",
          shim_version: "0.1.0",
          capabilities: ["shell", "files", "computer_use"],
          computer_use_driver: "cua-driver 0.38.0",
          computer_use_tools: ["click", "get_window_state", "type_text"],
        }),
      ),
    );

    render(<LocalPCBadge sessionID={SESSION_ID} />);

    await user.click(
      await screen.findByRole("button", {
        name: /local pc connected: macos arm64/i,
      }),
    );

    expect(
      await screen.findByText(
        /shell commands are not limited by that file root/i,
      ),
    ).toBeDefined();
    expect(screen.getByText(/FILE_\*/)).toBeDefined();
    expect(
      screen.getByText(/computer use: cua-driver 0\.38\.0, 3 tools/i),
    ).toBeDefined();
  });

  test("shows a distinct loading status", () => {
    server.use(
      http.get(EXECUTOR_URL, async () => {
        await delay("infinite");
        return HttpResponse.json({ kind: "none" });
      }),
    );

    render(<LocalPCBadge sessionID={SESSION_ID} />);

    expect(
      screen.getByRole("button", { name: /checking local pc/i }),
    ).toBeDefined();
  });

  test("announces when the status request fails", async () => {
    server.use(getGetExperimentalGetSessionExecutorMockHandler401());

    render(<LocalPCBadge sessionID={SESSION_ID} />);

    expect(
      await screen.findByRole("button", {
        name: /local pc status unavailable/i,
      }),
    ).toBeDefined();
  });

  test("does not show retained connection details after polling fails", async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({ kind: "shim", platform: "darwin" }),
      ),
    );
    render(
      <QueryClientProvider client={queryClient}>
        <LocalPCBadge sessionID={SESSION_ID} />
      </QueryClientProvider>,
    );
    await user.click(
      await screen.findByRole("button", { name: /local pc connected/i }),
    );
    server.use(getGetExperimentalGetSessionExecutorMockHandler401());
    await act(async () => {
      await queryClient.refetchQueries({
        queryKey: getGetExperimentalGetSessionExecutorQueryKey(SESSION_ID),
      });
    });

    expect(
      await screen.findByText(/status could not be loaded/i),
    ).toBeDefined();
    expect(screen.queryByText(/Files and commands route to/i)).toBeNull();
  });
});

describe("LocalPCBadge and the Local PC rollout", () => {
  const statusRequests = vi.fn();
  let client: QueryClient;

  async function advance(milliseconds: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(milliseconds);
    });
  }

  async function settled() {
    await act(async () => {
      await vi.waitFor(() => expect(client.isFetching()).toBe(0));
      await vi.advanceTimersByTimeAsync(1);
    });
  }

  function renderBadge() {
    render(
      <QueryClientProvider client={client}>
        <LocalPCBadge sessionID={SESSION_ID} machineID="machine-123456789" />
      </QueryClientProvider>,
    );
  }

  beforeEach(() => {
    vi.useFakeTimers();
    focusManager.setFocused(true);
    statusRequests.mockReset();
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    server.use(
      http.get(EXECUTOR_URL, () => {
        statusRequests();
        return HttpResponse.json({ kind: "shim", platform: "darwin" });
      }),
    );
  });

  afterEach(() => {
    client.clear();
    focusManager.setFocused(undefined);
    vi.useRealTimers();
  });

  test("asks the status route nothing for a user the flag service puts outside the rollout", async () => {
    serveRolloutFlag({ [Flag.LOCAL_PC_EXECUTOR]: false });

    renderBadge();
    await settled();
    await advance(60_000);

    expect(statusRequests).not.toHaveBeenCalled();
    fireEvent.click(
      screen.getByRole("button", {
        name: "Local PC is off. Open Local PC details",
      }),
    );
    expect(
      screen.getByText(
        "Local PC is turned off for your account, so this chat can't reach your computer.",
      ),
    ).toBeDefined();
    // Restarting the executor would not bring this chat back.
    expect(screen.queryByText(/restart autogpt-shim/i)).toBeNull();
  });

  test("keeps the status of a chat on the person's computer through a flag outage", async () => {
    serveRolloutFlag({});

    renderBadge();
    await settled();

    expect(statusRequests).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /^checking local pc/i }),
    ).toBeDefined();

    await advance(5_000);
    await settled();

    expect(statusRequests).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: /^local pc connected: macos/i }),
    ).toBeDefined();

    await advance(15_000);
    await settled();

    expect(statusRequests).toHaveBeenCalledTimes(2);
  });

  test("polls the status every 15 s for a user inside the rollout", async () => {
    serveRolloutFlag({ [Flag.LOCAL_PC_EXECUTOR]: true });

    renderBadge();
    await settled();
    expect(statusRequests).toHaveBeenCalledTimes(1);

    await advance(15_000);
    await settled();
    expect(statusRequests).toHaveBeenCalledTimes(2);
  });
});
