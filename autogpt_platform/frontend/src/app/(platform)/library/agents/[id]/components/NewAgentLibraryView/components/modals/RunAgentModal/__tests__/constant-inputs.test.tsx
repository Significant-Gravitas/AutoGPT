import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { render, screen } from "@/tests/integrations/test-utils";
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
  expect(screen.getByText("Events")).toBeTruthy();

  // ...and the graph's regular inputs are now surfaced under "Task Inputs"
  // (previously omitted for triggered agents).
  expect(screen.getByText("Task Inputs")).toBeTruthy();
  expect(screen.getByText("Topic")).toBeTruthy();
});
