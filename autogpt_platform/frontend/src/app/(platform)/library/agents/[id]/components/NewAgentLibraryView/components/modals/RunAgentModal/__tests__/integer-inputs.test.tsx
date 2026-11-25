import {
  getPostV1ExecuteGraphAgentMockHandler200,
  getPostV1ExecuteGraphAgentResponseMock200,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
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

function agentWithIntegerInputs() {
  return getGetV2GetLibraryAgentResponseMock({
    name: "Order Lookup",
    has_external_trigger: false,
    has_sensitive_action: false,
    has_human_in_the_loop: false,
    trigger_setup_info: null,
    credentials_input_schema: { properties: {}, required: [] },
    input_schema: {
      type: "object",
      properties: {
        order_id: { type: "integer", title: "Order ID" },
        offset: {
          anyOf: [{ type: "integer" }, { type: "null" }],
          title: "Offset",
        },
        limit: { type: "integer", title: "Limit" },
      },
      required: ["order_id"],
    },
  });
}

test("runs with an integer beyond 2^53 exactly as it was typed", async () => {
  let body: { inputs?: unknown } | null = null;
  server.use(
    getPostV1ExecuteGraphAgentMockHandler200(async (info) => {
      body = (await info.request.json()) as { inputs?: unknown };
      return getPostV1ExecuteGraphAgentResponseMock200();
    }),
  );

  const user = userEvent.setup();
  render(
    <RunAgentModal
      agent={agentWithIntegerInputs()}
      triggerSlot={<button>Open</button>}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Open" }));

  for (const [key, text] of [
    ["order_id", "9007199254740993"],
    ["offset", "-9223372036854775807"],
    ["limit", "42"],
  ]) {
    await expectTypingKept(
      (await screen.findByTestId(`agent-input-${key}`)) as HTMLInputElement,
      text,
    );
  }
  await user.click(screen.getByRole("button", { name: "Start Task" }));

  await waitFor(() => expect(body).not.toBeNull());
  expect(body!.inputs).toEqual({
    order_id: "9007199254740993",
    offset: "-9223372036854775807",
    limit: 42,
  });
});
