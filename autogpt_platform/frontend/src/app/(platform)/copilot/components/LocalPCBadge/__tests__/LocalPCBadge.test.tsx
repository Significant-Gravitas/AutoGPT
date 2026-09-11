import {
  getGetExperimentalGetSessionExecutorMockHandler200,
  getGetExperimentalGetSessionExecutorMockHandler401,
} from "@/app/api/__generated__/endpoints/copilot/copilot.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { describe, expect, test } from "vitest";
import { LocalPCBadge } from "../LocalPCBadge";

const EXECUTOR_URL =
  "http://localhost:3000/api/proxy/api/copilot/sessions/:sessionId/executor";
const SESSION_ID = "11111111-1111-4111-8111-111111111111";

describe("LocalPCBadge", () => {
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
          computer_use_features: [],
          computer_use_features_coarse: ["screenshot", "input"],
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
    expect(screen.getByText(/computer-use: screenshot, input/i)).toBeDefined();
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
});
