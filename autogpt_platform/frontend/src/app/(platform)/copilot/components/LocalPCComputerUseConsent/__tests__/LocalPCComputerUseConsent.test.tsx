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
const DRIVER = "cua-driver 0.38.0";
const READ_TOOLS = ["get_window_state", "list_windows"];
const INPUT_TOOLS = ["click", "type_text"];

function executorStatus(overrides: Record<string, unknown> = {}) {
  return {
    kind: "shim",
    machine_id: "machine-1",
    platform: "darwin",
    capabilities: ["computer_use"],
    computer_use_driver: DRIVER,
    computer_use_tools: [...READ_TOOLS, ...INPUT_TOOLS],
    computer_use_consent: "pending",
    ...overrides,
  };
}

function useExecutorHandler(
  consent: "pending" | "approved" | "denied" = "pending",
) {
  const requested = vi.fn();
  server.use(
    http.get(EXECUTOR_URL, () => {
      requested();
      return HttpResponse.json(
        executorStatus({ computer_use_consent: consent }),
      );
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
          HttpResponse.json(
            executorStatus({
              computer_use_tools: ["type_text", "click", "list_windows"],
              computer_use_consent: consentState,
            }),
          ),
        ),
        http.post(CONSENT_URL, async ({ request }) => {
          const body = await request.json();
          received(body);
          consentState = approved ? "approved" : "denied";
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
                expected_driver: DRIVER,
                expected_tools: ["click", "list_windows", "type_text"],
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

  test("names the driver doing the work", async () => {
    useExecutorHandler();

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(await screen.findByText(DRIVER)).toBeDefined();
  });

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
        return HttpResponse.json(
          executorStatus({
            machine_id: statusRequests === 1 ? "machine-a" : "machine-b",
            computer_use_tools:
              statusRequests === 1
                ? READ_TOOLS
                : [...READ_TOOLS, ...INPUT_TOOLS],
          }),
        );
      }),
      http.post(CONSENT_URL, () =>
        HttpResponse.json({ computer_use_consent: "approved" }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);
    expect(await screen.findByText(/see your open apps/i)).toBeDefined();
    expect(screen.queryByText(/click, type/i)).toBeNull();

    fireEvent.click(
      screen.getByRole("button", { name: "Allow for this session" }),
    );

    expect(await screen.findByText(/click, type/i)).toBeDefined();
    expect(screen.getByRole("dialog")).toBeDefined();
  });

  test("does not prompt when the shim lacks computer-use capability", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["files", "shell"],
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
        HttpResponse.json(
          executorStatus({
            computer_use_tools: [
              "check_permissions",
              "clipboard_read",
              "get_guide",
              "get_window_state",
              "launch_app",
            ],
          }),
        ),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText(
        "See your open apps and windows, read what they show, and take screenshots of them",
      ),
    ).toBeDefined();
    expect(screen.getByText("Open apps and links")).toBeDefined();
    expect(
      screen.getByText(
        "Read your clipboard, within the limits set on that computer",
      ),
    ).toBeDefined();
    for (const notGranted of [
      "Click, type, scroll, and use menus in your apps",
      "Put text on your clipboard, or a file from your computer once you approve it",
      "Give web pages files from your computer to upload, once you approve each one",
      "Move, resize, and bring forward your windows",
      "Control web pages in a browser it opens with its own profile",
      "Use other computer controls your Local PC offers",
    ]) {
      expect(screen.queryByText(notGranted)).toBeNull();
    }
  });

  test("says a file from the computer waits for the user's approval", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json(
          executorStatus({
            computer_use_tools: ["browser_set_input_files", "clipboard_write"],
          }),
        ),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText(
        "Give web pages files from your computer to upload, once you approve each one",
      ),
    ).toBeDefined();
    expect(
      screen.getByText(
        "Put text on your clipboard, or a file from your computer once you approve it",
      ),
    ).toBeDefined();
    expect(screen.getByText(/it asks you in this chat/)).toBeDefined();
  });

  test("still discloses a granted tool it has no wording for", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json(
          executorStatus({ computer_use_tools: ["click", "future_tool"] }),
        ),
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
    let tools = READ_TOOLS;
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json(
          executorStatus({ machine_id: machineID, computer_use_tools: tools }),
        ),
      ),
      http.post(CONSENT_URL, async ({ request }) => {
        expect(await request.json()).toEqual({
          approved: true,
          expected_machine_id: "machine-a",
          expected_driver: DRIVER,
          expected_tools: READ_TOOLS,
        });
        machineID = "machine-b";
        tools = [...READ_TOOLS, ...INPUT_TOOLS];
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
    expect(await screen.findByText(/click, type/i)).toBeDefined();
  });

  test("does not prompt when nothing would be granted", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json(
          executorStatus({
            computer_use_driver: null,
            computer_use_tools: null,
          }),
        ),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});
