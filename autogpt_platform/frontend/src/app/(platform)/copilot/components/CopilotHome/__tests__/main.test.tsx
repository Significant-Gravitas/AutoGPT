import { expect, test, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetBriefingsGetLatestBriefingMockHandler200 } from "@/app/api/__generated__/endpoints/briefings/briefings.msw";
import { CopilotHome } from "../CopilotHome";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.BRIEFING_HOME || flag === actual.Flag.HIRE_EXPERTS,
  };
});

const baseProps = {
  inputLayoutId: "test-layout",
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
};

// Guarantees the pulse strip has at least one chip regardless of faker's
// random defaults — an agent with an external trigger always produces a
// "listening" sitrep item.
function mockPulseStripAgent() {
  const base = getGetV2ListLibraryAgentsResponseMock();
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...base,
      agents: [
        { ...base.agents[0], graph_id: "g-1", has_external_trigger: true },
      ],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
}

test("renders greeting and composer", async () => {
  mockPulseStripAgent();
  render(<CopilotHome {...baseProps} />);
  expect(await screen.findByPlaceholderText(/./)).toBeDefined();
});

test("falls back to pulse strip when there is no briefing", async () => {
  mockPulseStripAgent();
  server.use(getGetBriefingsGetLatestBriefingMockHandler200(null));
  render(<CopilotHome {...baseProps} />);
  expect(
    await screen.findByText("What's happening with your agents"),
  ).toBeDefined();
});
