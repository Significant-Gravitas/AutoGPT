import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import {
  getGetV1ListAllExecutionsMockHandler,
  getPostV1CreateNewGraphMockHandler,
  getPostV1CreateNewGraphResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { CreateGraph } from "@/app/api/__generated__/models/createGraph";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import LibraryPage from "../page";

vi.mock("posthog-js", () => ({ default: { capture: vi.fn() } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "import-file-user", created_at: "2026-10-01T00:00:00Z" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => ({ state: null, completeStep: vi.fn() }),
  default: ({ children }: { children: ReactNode }) => children,
}));

// What an export of a workflow written in French and German holds.
const WORKFLOW = {
  name: "Résumé des ventes ✓",
  description: "Écrit une note — pour l’équipe",
  nodes: [
    {
      id: "compose",
      block_id: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
      input_default: { format: "Grüße, {{ name }} « {{ topic }} »" },
      metadata: { customized_name: "Compose the note ✎" },
      input_links: [],
      output_links: [],
    },
  ],
  links: [],
};

const originalLocation = window.location;
let navigations: string[] = [];

// A finished import sends the browser to the builder. The page stays put, so
// the code that reads its URL (the query state) keeps a valid one.
function stubLocation() {
  const url = new URL("http://localhost/library");
  navigations = [];
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: {
      origin: url.origin,
      protocol: url.protocol,
      host: url.host,
      hostname: url.hostname,
      port: url.port,
      pathname: url.pathname,
      search: url.search,
      hash: url.hash,
      get href() {
        return url.href;
      },
      set href(next: string) {
        navigations.push(next);
      },
    },
  });
}

beforeEach(() => {
  const pagination = {
    total_items: 0,
    total_pages: 1,
    current_page: 1,
    page_size: 20,
  };
  server.use(
    getGetV2ListLibraryAgentsMockHandler({ agents: [], pagination }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({ agents: [], pagination }),
    getGetV2ListLibraryFoldersMockHandler({ folders: [], pagination }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
  stubLocation();
});

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: originalLocation,
  });
});

function captureCreatedGraph() {
  const sent = vi.fn<(body: CreateGraph) => void>();
  server.use(
    getPostV1CreateNewGraphMockHandler(async ({ request }) => {
      sent((await request.json()) as CreateGraph);
      return { ...getPostV1CreateNewGraphResponseMock(), id: "imported" };
    }),
  );
  return sent;
}

async function uploadWorkflowFile(text: string) {
  render(<LibraryPage />);

  await userEvent.click(screen.getAllByTestId("import-button")[0]);
  const dialog = await screen.findByRole("dialog", { name: "Import" });
  await userEvent.click(
    within(dialog).getByRole("tab", { name: "AutoGPT workflow" }),
  );
  const fileInput =
    dialog.querySelector<HTMLInputElement>('input[type="file"]');
  if (!fileInput) throw new Error("The AutoGPT workflow tab has no file input");
  await userEvent.upload(
    fileInput,
    new File([text], "workflow.json", { type: "application/json" }),
  );
  return dialog;
}

test.each([
  { file: "as exported", text: JSON.stringify(WORKFLOW, null, 2) },
  // Some editors save UTF-8 files with a byte-order mark.
  {
    file: "saved with a byte-order mark",
    text: `\uFEFF${JSON.stringify(WORKFLOW)}`,
  },
])(
  "a workflow file in any language imports as written ($file)",
  async ({ text }) => {
    const sent = captureCreatedGraph();
    const dialog = await uploadWorkflowFile(text);

    // The dialog fills in the name and description from the file it read.
    const name = within(dialog).getByRole("textbox", { name: "Workflow name" });
    await waitFor(() => expect(name).toHaveProperty("value", WORKFLOW.name));
    expect(
      within(dialog).getByRole("textbox", { name: "Workflow description" }),
    ).toHaveProperty("value", WORKFLOW.description);

    await userEvent.click(
      within(dialog).getByRole("button", { name: "Upload" }),
    );
    await waitFor(() => expect(sent).toHaveBeenCalledOnce());
    expect(sent.mock.calls[0][0]).toMatchObject({
      source: "upload",
      graph: {
        name: WORKFLOW.name,
        description: WORKFLOW.description,
        nodes: [
          {
            input_default: WORKFLOW.nodes[0].input_default,
            metadata: WORKFLOW.nodes[0].metadata,
          },
        ],
      },
    });
    await waitFor(() =>
      expect(navigations).toEqual(["/build?flowID=imported"]),
    );
  },
);

test('an imported workflow drops the exporter\'s credentials and keeps data keyed "credentials"', async () => {
  const exporters = (provider: string) => ({
    id: `exporters-${provider}-key`,
    provider,
    type: "api_key",
  });
  const file = {
    name: "Support triage",
    description: "Answers and fixes support tickets",
    nodes: [
      {
        id: "fix",
        block_id: "4e34f4a5-9b89-4326-ba77-2dd6750b7194",
        input_default: {
          prompt: "Fix the failing test",
          e2b_credentials: exporters("e2b"),
          anthropic_credentials: exporters("anthropic"),
        },
      },
      {
        id: "compose",
        block_id: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
        input_default: {
          format: "Sign in as {{credentials}}",
          values: { credentials: "the on-call login" },
        },
      },
    ],
    links: [],
  };
  const sent = captureCreatedGraph();
  const dialog = await uploadWorkflowFile(JSON.stringify(file));

  const name = within(dialog).getByRole("textbox", { name: "Workflow name" });
  await waitFor(() => expect(name).toHaveProperty("value", file.name));
  await userEvent.click(within(dialog).getByRole("button", { name: "Upload" }));

  await waitFor(() => expect(sent).toHaveBeenCalledOnce());
  const imported = sent.mock.calls[0][0].graph.nodes ?? [];
  expect(imported.map((node) => node.input_default)).toEqual([
    { prompt: "Fix the failing test" },
    {
      format: "Sign in as {{credentials}}",
      values: { credentials: "the on-call login" },
    },
  ]);
});
