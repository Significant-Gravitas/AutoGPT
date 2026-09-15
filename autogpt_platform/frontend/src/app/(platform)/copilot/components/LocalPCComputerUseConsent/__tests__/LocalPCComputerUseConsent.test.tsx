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
        screen.queryByText(/claude is requesting computer access/i),
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

  test("discloses the union of coarse and fine-grained capabilities", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_features_coarse: ["screenshot", "apps"],
          computer_use_features: ["input.click", "clipboard.read"],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText(/capture screenshots of your screen/i),
    ).toBeDefined();
    expect(screen.getByText(/list and launch apps/i)).toBeDefined();
    expect(screen.getByText(/move the pointer/i)).toBeDefined();
    expect(screen.queryByText(/open windows/i)).toBeNull();
    expect(screen.getByText(/read and write your clipboard/i)).toBeDefined();
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

  test("recognizes dotted clipboard capabilities from older shims", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_features: ["clipboard.read", "clipboard.write"],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    expect(
      await screen.findByText(/read and write your clipboard/i),
    ).toBeDefined();
  });

  test("ignores unknown coarse capability names", async () => {
    server.use(
      http.get(EXECUTOR_URL, () =>
        HttpResponse.json({
          kind: "shim",
          capabilities: ["computer_use"],
          computer_use_features_coarse: ["future-unknown-capability"],
          computer_use_features: [],
          computer_use_consent: "pending",
        }),
      ),
    );

    render(<LocalPCComputerUseConsent sessionID="test-session-id" />);

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});
