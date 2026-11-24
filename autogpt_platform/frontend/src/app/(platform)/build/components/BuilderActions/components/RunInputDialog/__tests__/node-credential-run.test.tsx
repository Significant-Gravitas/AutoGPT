import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { server } from "@/mocks/mock-server";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { ReactFlowProvider } from "@xyflow/react";
import { HttpResponse, http } from "msw";
import { afterEach, expect, test, vi } from "vitest";
import { RunInputDialog } from "../RunInputDialog";

// CredentialsProvider only loads credentials for a signed-in user.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

const FIELD = "firecrawl_api_key_credentials";
const PLATFORM_KEY = {
  id: "firecrawl-default",
  provider: "firecrawl",
  type: "api_key",
  title: "Firecrawl API Key",
  is_system: true,
};
const USER_KEY = {
  id: "user-firecrawl-key",
  provider: "firecrawl",
  type: "api_key",
  title: "My Firecrawl key",
  is_system: false,
};

function scrapeStep(hardcodedValues: Record<string, unknown>) {
  return {
    id: "scrape-step",
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues,
      title: "Firecrawl Scrape",
      description: "Scrape a page",
      inputSchema: {},
      outputSchema: {},
      block_id: "firecrawl-scrape",
      costs: [],
      categories: [],
    },
  } as unknown as CustomNode;
}

function mockBuilderRun(stepValues: Record<string, unknown>) {
  const executeBodies: Array<{ credentials_inputs?: unknown }> = [];
  useGraphStore.setState({
    inputSchema: null,
    credentialsInputSchema: {
      type: "object",
      properties: {
        [FIELD]: {
          credentials_provider: ["firecrawl"],
          credentials_types: ["api_key"],
          title: "Credentials",
          type: "object",
        },
      },
      required: [FIELD],
    },
  });
  useNodeStore.setState({ nodes: [scrapeStep(stepValues)] });
  server.use(
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([
        { name: "firecrawl", description: "Web scraping and crawling" },
      ]),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(["firecrawl"]),
    ),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json([PLATFORM_KEY, USER_KEY]),
    ),
    getPostV1ExecuteGraphAgentMockHandler(async ({ request }) => {
      executeBodies.push((await request.json()) as never);
      return getPostV1ExecuteGraphAgentResponseMock({
        id: "builder-run",
        graph_id: "graph-1",
      });
    }),
  );
  return executeBodies;
}

function renderRunDialog() {
  render(
    <CredentialsProvider>
      <ReactFlowProvider>
        <RunInputDialog
          isOpen
          setIsOpen={vi.fn()}
          purpose="run"
          graphID="graph-1"
          graphVersion={1}
        />
      </ReactFlowProvider>
    </CredentialsProvider>,
  );
}

function keyIn(credentials: unknown) {
  return (credentials as Record<string, { id?: string }> | undefined)?.[FIELD]
    ?.id;
}

afterEach(() => {
  useGraphStore.getState().reset();
  useNodeStore.setState({ nodes: [] });
});

test("a builder run uses the key picked on the step, and shows it", async () => {
  const executeBodies = mockBuilderRun({
    url: "https://example.com",
    credentials: {
      id: USER_KEY.id,
      provider: "firecrawl",
      type: "api_key",
      title: USER_KEY.title,
    },
  });
  renderRunDialog();

  const picker = await screen.findByRole("combobox", {
    name: /firecrawl credential/i,
  });
  await waitFor(() =>
    expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id),
  );
  await userEvent.click(screen.getByRole("button", { name: "Manual Run" }));

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("a builder run whose step has no key of its own uses the platform key, as before", async () => {
  const executeBodies = mockBuilderRun({ url: "https://example.com" });
  renderRunDialog();

  // The platform key is picked inside the collapsed section, as before
  await userEvent.click(
    await screen.findByRole("button", { name: /system credentials/i }),
  );
  const picker = await screen.findByRole("combobox", {
    name: /firecrawl credential/i,
  });
  await waitFor(() =>
    expect((picker as HTMLSelectElement).value).toBe(PLATFORM_KEY.id),
  );
  await userEvent.click(screen.getByRole("button", { name: "Manual Run" }));

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
});
