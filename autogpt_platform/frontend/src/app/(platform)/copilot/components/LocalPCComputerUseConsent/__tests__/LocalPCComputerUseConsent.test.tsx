import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import { LocalPCComputerUseConsent } from "../LocalPCComputerUseConsent";

const EXECUTOR_URL =
  "http://localhost:3000/api/proxy/api/copilot/sessions/:sessionId/executor";
const CONSENT_URL = `${EXECUTOR_URL}/consent`;

const COARSE_TOOLS: Record<string, string[]> = {
  screenshot: ["local_pc_screenshot"],
  input: [
    "local_pc_click",
    "local_pc_type",
    "local_pc_key",
    "local_pc_scroll",
    "local_pc_cursor_position",
  ],
};

function toolsFor(coarse: string[]) {
  return coarse.flatMap((feature) => COARSE_TOOLS[feature] ?? []);
}

function useExecutorHandler(
  consent: "pending" | "approved" | "denied" = "pending",
) {
  const requested = vi.fn();
  server.use(
    http.get(EXECUTOR_URL, () => {
      requested();
      return HttpResponse.json({
        kind: "shim",
        machine_id: "machine-1",
        platform: "darwin",
        capabilities: ["computer_use"],
        computer_use_features_coarse: ["screenshot", "input"],
        computer_use_features: [],
        computer_use_tools: toolsFor(["screenshot", "input"]),
        computer_use_consent: consent,
      });
    }),
  );
  return requested;
}

describe("LocalPCComputerUseConsent", () => {
  test.each([
    ["Not this time", false, "denied"],
    ["Allow for this session", true, "approved"],
  ] as const)(
    "%s persists the decision to the session-scoped backend gate",
    async (buttonName, approved, resultState) => {
      let consentState: "pending" | "approved" | "denied" = "pending";
      const received = vi.fn();
      server.use(
        http.get(EXECUTOR_URL, () =>
          HttpResponse.json({
            kind: "shim",
            machine_id: "machine-1",
            platform: "darwin",
            capabilities: ["computer_use"],
            computer_use_features_coarse: ["screenshot", "input"],
            computer_use_features: [],
            computer_use_tools: toolsFor(["screenshot", "input"]),
            computer_use_consent: consentState,
          }),
        ),
        http.post(CONSENT_URL, async ({ request }) => {
          const body = (await request.json()) as {
            approved: boolean;
            expected_machine_id?: string;
            expected_features_coarse?: string[];
            expected_features?: string[];
          };
          received(body);
          consentState = body.approved ? "approved" : "denied";
          return HttpResponse.json({ computer_use_consent: consentState });
        }),
      );

      render(<LocalPCComputerUseConsent sessionID="test-session-id" />);
      fireEvent.click(await screen.findByRole("button", { name: buttonName }));

      await waitFor(() =>
        expect(received).toHaveBeenCalledWith(
          approved
            ? {
                approved: true,
                expected_machine_id: "machine-1",
                expected_features_coarse: ["input", "screenshot"],
                expected_features: [],
              }
            : { approved: false },
        ),
      );
      expect(consentState).toBe(resultState);
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    },
  );

  test.each(["approved", "denied"] as const)(
    "does not prompt when server-side consent is already %s",
    async (consent) => {
      const requested = useExecutorHandler(consent);

      render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

      await waitFor(() => expect(requested).toHaveBeenCalledOnce());
      expect(
        screen.queryByText(/otto is requesting computer access/i),
      ).toBeNull();
    },
  );

  test("keeps the decision dialog open when the backend rejects the update", async () => {
    useExecutorHandler();
    server.use(
      http.post(CONSENT_URL, () =>
        HttpResponse.json({ detail: "executor offline" }, { status: 503 }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);
    expect(
      await screen.findByText("Otto is requesting computer access"),
    ).toBeDefined();
    fireEvent.click(
      await screen.findByRole("button", { name: "Not this time" }),
    );

    expect((await screen.findByRole("alert")).textContent).toMatch(
      /could not update computer access/i,
    );
    expect(screen.getByRole("dialog")).toBeDefined();
  });

  test("prompts again when the same session reconnects with broader access", async () => {
    let statusRequests = 0;
    server.use(
      http.get(EXECUTOR_URL, () => {
        statusRequests += 1;
        return HttpResponse.json({
          kind: "shim",
          machine_id: statusRequests === 1 ? "machine-a" : "machine-b",
          capabilities: ["computer_use"],
          computer_use_features_coarse:
            statusRequests === 1 ? ["screenshot"] : ["screenshot", "input"],
          computer_use_features: [],
          computer_use_tools: toolsFor(
            statusRequests === 1 ? ["screenshot"] : ["screenshot", "input"],
          ),
          computer_use_consent: "pending",
        });
      }),
      http.post(CONSENT_URL, () =>
        HttpResponse.json({ computer_use_consent: "approved" }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);
    expect(
      await screen.findByText(/capture screenshots of your screen/i),
    ).toBeDefined();
    expect(screen.queryByText(/move the pointer/i)).toBeNull();

    fireEvent.click(
      screen.getByRole("button", { name: "Allow for this session" }),
    );

    expect(await screen.findByText(/move the pointer/i)).toBeDefined();
    expect(screen.getByRole("dialog")).toBeDefined();
  });

  test("does not prompt when the shim lacks computer-use capability", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["files", "shell"],
          computer_use_features: [],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  test("describes only the actions approving would grant", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_features_coarse: ["screenshot", "apps"],
          computer_use_features: ["input.click", "clipboard.read"],
          computer_use_tools: [
            "local_pc_click",
            "local_pc_clipboard_read",
            "local_pc_launch_app",
            "local_pc_list_apps",
            "local_pc_screenshot",
          ],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText("Capture screenshots of your screen"),
    ).toBeDefined();
    expect(screen.getByText("Move the pointer and click")).toBeDefined();
    expect(screen.getByText("List and launch apps")).toBeDefined();
    expect(screen.getByText("Read your clipboard")).toBeDefined();
    for (const notGranted of [
      "Move the pointer, click, type, scroll, and send keys",
      "Type text",
      "Send keys and shortcuts",
      "Read and write your clipboard",
      "Write to your clipboard",
      "List and focus open windows",
    ]) {
      expect(screen.queryByText(notGranted)).toBeNull();
    }
  });

  test("still discloses a granted tool it has no wording for", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_tools: ["local_pc_screenshot", "local_pc_future_tool"],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText(
        "Use other computer controls your Local PC offers",
      ),
    ).toBeDefined();
  });

  test("refetches and re-prompts when approval scope changes before submission", async () => {
    let machineID = "machine-a";
    let features = ["screenshot"];
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          machine_id: machineID,
          capabilities: ["computer_use"],
          computer_use_features_coarse: features,
          computer_use_features: [],
          computer_use_tools: toolsFor(features),
          computer_use_consent: "pending",
        }),
      ),
      http.post(CONSENT_URL, async ({ request }) => {
        expect(await request.json()).toEqual({
          approved: true,
          expected_machine_id: "machine-a",
          expected_features_coarse: ["screenshot"],
          expected_features: [],
        });
        machineID = "machine-b";
        features = ["screenshot", "input"];
        return HttpResponse.json(
          { detail: "Local PC executor scope changed" },
          { status: 409 },
        );
      }),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Allow for this session" }),
    );

    expect(await screen.findByRole("alert")).toBeDefined();
    expect(await screen.findByText(/move the pointer/i)).toBeDefined();
  });

  test("describes a fully granted area in one line", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_features: ["clipboard.read", "clipboard.write"],
          computer_use_tools: [
            "local_pc_clipboard_read",
            "local_pc_clipboard_write",
          ],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText(/read and write your clipboard/i),
    ).toBeDefined();
  });

  test("does not prompt when nothing would be granted", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_features_coarse: ["future-unknown-capability"],
          computer_use_features: [],
          computer_use_tools: null,
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});
