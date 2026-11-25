import {
  getPatchV1UpdateGraphSettingsMockHandler,
  getPatchV1UpdateGraphSettingsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { AgentSettingsModal } from "../AgentSettingsModal";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn(), toasts: [], dismiss: vi.fn() }),
  toast: vi.fn(),
  useToastOnFail: () => vi.fn(),
}));

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

const agent = getGetV2GetLibraryAgentResponseMock({
  id: "library-agent",
  graph_id: "graph-1",
  has_human_in_the_loop: true,
  has_sensitive_action: true,
  settings: {
    human_in_the_loop_safe_mode: true,
    sensitive_action_safe_mode: false,
  },
});

test("names each safe-mode switch after its setting", () => {
  render(
    <AgentSettingsModal agent={agent} controlledOpen onOpenChange={vi.fn()} />,
  );

  const humanInTheLoop = screen.getByRole("switch", {
    name: "Human-in-the-loop approval",
    description:
      "The workflow will pause at human-in-the-loop blocks and wait for your review before continuing",
  });
  const sensitiveActions = screen.getByRole("switch", {
    name: "Sensitive action approval",
    description:
      "The workflow will pause at sensitive action blocks and wait for your review before continuing",
  });

  expect(humanInTheLoop.getAttribute("aria-checked")).toBe("true");
  expect(sensitiveActions.getAttribute("aria-checked")).toBe("false");
});

test.each([
  {
    name: "Human-in-the-loop approval",
    saves: { human_in_the_loop_safe_mode: false },
  },
  {
    name: "Sensitive action approval",
    saves: { sensitive_action_safe_mode: true },
  },
])("the $name switch saves its own setting", async ({ name, saves }) => {
  const saved = vi.fn();
  server.use(
    getPatchV1UpdateGraphSettingsMockHandler(async ({ request, params }) => {
      saved(params.graphId, await request.json());
      return getPatchV1UpdateGraphSettingsResponseMock(saves);
    }),
  );
  const user = userEvent.setup();
  render(
    <AgentSettingsModal agent={agent} controlledOpen onOpenChange={vi.fn()} />,
  );

  await user.click(screen.getByRole("switch", { name }));

  await waitFor(() => expect(saved).toHaveBeenCalledWith("graph-1", saves));
  expect(saved).toHaveBeenCalledOnce();
});
