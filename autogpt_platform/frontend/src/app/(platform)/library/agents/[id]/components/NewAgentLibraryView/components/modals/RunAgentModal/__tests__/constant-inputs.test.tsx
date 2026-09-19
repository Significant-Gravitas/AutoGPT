import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getPostV2SetupTriggerMockHandler200,
  getPostV2SetupTriggerResponseMock,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeAll, expect, test, vi } from "vitest";
import { RunAgentModal } from "../RunAgentModal";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn(), toasts: [], dismiss: vi.fn() }),
  toast: vi.fn(),
  useToastOnFail: () => vi.fn(),
}));

beforeAll(() => {
  if (!globalThis.ResizeObserver) {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

// A triggered agent that ALSO has a regular input node. The setup modal must
// surface both the trigger config (sent as `trigger_config`) and the regular
// graph inputs (sent as `constant_inputs`) — not just the trigger config.
function triggeredAgentWithInputs() {
  return getGetV2GetLibraryAgentResponseMock({
    name: "Watcher",
    has_external_trigger: true,
    has_sensitive_action: false,
    has_human_in_the_loop: false,
    credentials_input_schema: { properties: {}, required: [] },
    trigger_setup_info: {
      provider: "github",
      credentials_input_name: null, // manual trigger -> no credentials step
      config_schema: {
        type: "object",
        properties: { events: { type: "string", title: "Events" } },
        required: [],
      },
    },
    input_schema: {
      type: "object",
      properties: { topic: { type: "string", title: "Topic" } },
      required: ["topic"],
    },
  });
}

test("setup modal surfaces regular graph inputs alongside the trigger config", async () => {
  const user = userEvent.setup();
  render(
    <RunAgentModal
      agent={triggeredAgentWithInputs()}
      triggerSlot={<button>Open</button>}
    />,
  );

  await user.click(screen.getByRole("button", { name: "Open" }));

  // Trigger config is shown under its own section...
  expect(await screen.findByText("Trigger Configuration")).toBeTruthy();
  expect(await screen.findByText("Events")).toBeTruthy();

  // ...and the graph's regular inputs are now surfaced under "Task Inputs"
  // (previously omitted for triggered agents).
  expect(await screen.findByText("Task Inputs")).toBeTruthy();
  expect(await screen.findByText("Topic")).toBeTruthy();
});

test("sends trigger config and regular inputs as distinct payload fields", async () => {
  let captured: Record<string, unknown> | null = null;
  server.use(
    getPostV2SetupTriggerMockHandler200(async (info) => {
      captured = (await info.request.json()) as Record<string, unknown>;
      return getPostV2SetupTriggerResponseMock();
    }),
  );

  const user = userEvent.setup();
  // Seed both groups via props (also exercises `initialTriggerConfigValues`),
  // so the test only needs to name the trigger and submit.
  render(
    <RunAgentModal
      agent={triggeredAgentWithInputs()}
      triggerSlot={<button>Open</button>}
      initialInputValues={{ topic: "weather" }}
      initialTriggerConfigValues={{ events: "push" }}
    />,
  );

  await user.click(screen.getByRole("button", { name: "Open" }));
  await screen.findByText("Trigger Configuration");

  await user.type(
    screen.getByPlaceholderText("Enter trigger name"),
    "My Trigger",
  );
  await user.click(screen.getByRole("button", { name: /set up trigger/i }));

  await waitFor(() => expect(captured).not.toBeNull());
  // The two groups must be sent as separate fields, not merged.
  expect(captured!.trigger_config).toEqual({ events: "push" });
  expect(captured!.constant_inputs).toEqual({ topic: "weather" });
});
