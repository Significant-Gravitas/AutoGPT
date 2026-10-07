import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { delay, http, HttpResponse } from "msw";
import { expect, fn, screen, userEvent } from "storybook/test";
import {
  getInstallExpertWorkflowMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListStoreAgentsMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { MANAGED_IDENTITIES } from "@/components/molecules/ExpertAvatar/helpers";
import { InstallWorkflowPicker } from "./InstallWorkflowPicker";

const FIXED_DATE = new Date("2026-08-12T09:00:00Z");
const TARGET_EXPERT_ID = "exp-maria";

function expert(overrides: Partial<Expert> = {}): Expert {
  return {
    id: TARGET_EXPERT_ID,
    name: "Maria",
    avatar_url: MANAGED_IDENTITIES[0].url,
    role: "Finance",
    tagline: null,
    bio: null,
    skills: [],
    identity: "",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    ...overrides,
  };
}

const EXPERTS: Expert[] = [
  expert({
    workflows: [
      {
        id: "wf-1",
        store_listing_version_id: null,
        library_agent_id: "lib-installed",
        graph_id: "g-installed",
        name: "Already installed",
        description: null,
      },
    ],
  }),
  expert({
    id: "exp-jules",
    name: "Jules",
    avatar_url: MANAGED_IDENTITIES[1].url,
    role: "Sales",
  }),
  expert({
    id: "exp-nadia",
    name: "Nadia",
    avatar_url: null,
    role: "Marketing",
  }),
  expert({ id: "exp-template", name: "Template", is_template: true }),
  expert({ id: "exp-archived", name: "Archived", is_archived: true }),
];

function libraryAgent(
  index: number,
  overrides: Partial<LibraryAgent> = {},
): LibraryAgent {
  return {
    id: `lib-${index}`,
    graph_id: `g-${index}`,
    graph_version: 1,
    image_url: null,
    creator_name: "Unknown",
    creator_image_url: "",
    status: "HEALTHY",
    created_at: FIXED_DATE,
    updated_at: FIXED_DATE,
    name: `Workflow ${index}`,
    description: "",
    input_schema: {},
    output_schema: {},
    credentials_input_schema: null,
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    new_output: false,
    can_access_graph: true,
    is_latest_version: true,
    is_favorite: false,
    ...overrides,
  };
}

const LIBRARY_AGENTS: LibraryAgent[] = [
  libraryAgent(1, {
    name: "Lead finder",
    description: "Finds new leads that match your ideal customer profile.",
    image_url: "/images/team-card-banner.jpg",
  }),
  libraryAgent(2, {
    name: "Invoice chaser",
    description: "Sends polite reminders for overdue invoices.",
  }),
  libraryAgent(3, { name: "Morning digest", description: "" }),
  libraryAgent(0, { id: "lib-installed", name: "Already installed" }),
];

function libraryPage(
  agents: LibraryAgent[],
  currentPage = 1,
  totalPages = 1,
): LibraryAgentResponse {
  return {
    agents,
    pagination: {
      total_items: agents.length * totalPages,
      total_pages: totalPages,
      current_page: currentPage,
      page_size: 10,
    },
  };
}

function storeAgent(
  index: number,
  overrides: Partial<StoreAgent> = {},
): StoreAgent {
  return {
    slug: `workflow-${index}`,
    agent_name: `Workflow ${index}`,
    agent_image: "",
    creator: "autogpt",
    creator_avatar: "",
    sub_heading: "",
    description: "",
    runs: 120,
    rating: 4.6,
    agent_graph_id: `store-g-${index}`,
    ...overrides,
  };
}

const STORE_AGENTS: StoreAgentsResponse = {
  agents: [
    storeAgent(1, {
      agent_name: "SEO blog writer",
      creator: "autogpt",
      agent_image: "/images/tour-og.png",
    }),
    storeAgent(2, { agent_name: "Meeting notes to tasks", creator: "ana" }),
    storeAgent(3, { agent_name: "Cold email personaliser", creator: "leo" }),
  ],
  pagination: {
    total_items: 3,
    total_pages: 1,
    current_page: 1,
    page_size: 10,
  },
};

const EMPTY_STORE: StoreAgentsResponse = {
  agents: [],
  pagination: {
    total_items: 0,
    total_pages: 0,
    current_page: 1,
    page_size: 10,
  },
};

const NEVER_INSTALL = http.post(
  "*/api/experts/:expertId/workflows",
  async () => {
    await delay("infinite");
    return HttpResponse.json({});
  },
);

const DEFAULT_HANDLERS = [
  getListExpertsMockHandler(EXPERTS),
  getGetV2ListLibraryAgentsMockHandler(libraryPage(LIBRARY_AGENTS)),
  getGetV2ListStoreAgentsMockHandler(STORE_AGENTS),
  getInstallExpertWorkflowMockHandler({
    id: "wf-new",
    store_listing_version_id: null,
    library_agent_id: "lib-1",
    graph_id: "g-1",
    name: "Lead finder",
    description: null,
  }),
];

function pendingGet(path: string) {
  return http.get(path, async () => {
    await delay("infinite");
    return HttpResponse.json({});
  });
}

function failingGet(path: string) {
  return http.get(path, () =>
    HttpResponse.json({ detail: "Internal server error" }, { status: 500 }),
  );
}

async function openMarketplace() {
  await userEvent.click(
    await screen.findByRole("button", { name: "Marketplace" }),
  );
}

const meta = {
  title: "Molecules/InstallWorkflowPicker",
  component: InstallWorkflowPicker,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    msw: { handlers: DEFAULT_HANDLERS },
    docs: {
      description: {
        component:
          "Dialog that installs a workflow on a hired expert. In `pick-workflow` mode (an expert's page) it lists the user's library workflows, or marketplace workflows via the source toggle, with debounced search and paging; workflows the expert already has are left out. In `pick-expert` mode (a marketplace listing) it lists hired experts to install that listing on. Data comes from `/api/experts`, `/api/library/agents` and `/api/store/agents`.",
      },
    },
  },
  args: {
    mode: "pick-workflow",
    expertId: TARGET_EXPERT_ID,
    open: true,
    onClose: fn(),
  },
} satisfies Meta<typeof InstallWorkflowPicker>;

export default meta;
type Story = StoryObj<typeof meta>;

export const LibraryWorkflows: Story = {
  play: async () => {
    await expect(await screen.findByText("Lead finder")).toBeInTheDocument();
  },
};

export const LibraryLoading: Story = {
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler(EXPERTS),
        pendingGet("*/api/library/agents"),
      ],
    },
  },
};

export const LibraryEmpty: Story = {
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler(EXPERTS),
        getGetV2ListLibraryAgentsMockHandler(libraryPage([], 1, 0)),
      ],
    },
  },
};

export const LibraryError: Story = {
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler(EXPERTS),
        failingGet("*/api/library/agents"),
      ],
    },
    docs: {
      description: {
        story:
          "A failed library request has no error state of its own: it reads as an empty library.",
      },
    },
  },
};

export const LibraryWithMorePages: Story = {
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler(EXPERTS),
        getGetV2ListLibraryAgentsMockHandler(({ request }) => {
          const page = Number(new URL(request.url).searchParams.get("page"));
          return page === 2
            ? libraryPage(
                [
                  libraryAgent(11, {
                    name: "Expense sorter",
                    description: "Files receipts under the right cost centre.",
                  }),
                ],
                2,
                2,
              )
            : libraryPage(LIBRARY_AGENTS.slice(0, 3), 1, 2);
        }),
      ],
    },
  },
};

export const SearchWithoutResults: Story = {
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler(EXPERTS),
        getGetV2ListLibraryAgentsMockHandler(({ request }) =>
          new URL(request.url).searchParams.get("search_term")
            ? libraryPage([], 1, 0)
            : libraryPage(LIBRARY_AGENTS),
        ),
      ],
    },
  },
  play: async () => {
    await userEvent.type(
      await screen.findByRole("textbox", { name: "Search workflows" }),
      "payroll",
    );
    await expect(
      await screen.findByText("No workflows in your library."),
    ).toBeInTheDocument();
  },
};

export const Installing: Story = {
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler(EXPERTS),
        getGetV2ListLibraryAgentsMockHandler(libraryPage(LIBRARY_AGENTS)),
        NEVER_INSTALL,
      ],
    },
  },
  play: async () => {
    const [firstInstall] = await screen.findAllByRole("button", {
      name: "Install",
    });
    await userEvent.click(firstInstall);
  },
};

export const MarketplaceWorkflows: Story = {
  play: async () => {
    await openMarketplace();
    await expect(
      await screen.findByText("SEO blog writer"),
    ).toBeInTheDocument();
  },
};

export const MarketplaceLoading: Story = {
  parameters: {
    msw: {
      handlers: [
        ...DEFAULT_HANDLERS.slice(0, 2),
        pendingGet("*/api/store/agents"),
      ],
    },
  },
  play: openMarketplace,
};

export const MarketplaceEmpty: Story = {
  parameters: {
    msw: {
      handlers: [
        ...DEFAULT_HANDLERS.slice(0, 2),
        getGetV2ListStoreAgentsMockHandler(EMPTY_STORE),
      ],
    },
  },
  play: openMarketplace,
};

export const MarketplaceError: Story = {
  parameters: {
    msw: {
      handlers: [
        ...DEFAULT_HANDLERS.slice(0, 2),
        failingGet("*/api/store/agents"),
      ],
    },
    docs: {
      description: {
        story:
          "A failed marketplace request also reads as “No workflows found.”",
      },
    },
  },
  play: openMarketplace,
};

export const PickExpert: Story = {
  args: {
    mode: "pick-expert",
    expertId: undefined,
    storeListingVersionId: "slv-seo-blog-writer",
  },
  parameters: {
    docs: {
      description: {
        story:
          "From a marketplace listing: the hired experts (templates and archived experts are left out).",
      },
    },
  },
};

export const PickExpertInstalling: Story = {
  args: PickExpert.args,
  parameters: {
    msw: { handlers: [getListExpertsMockHandler(EXPERTS), NEVER_INSTALL] },
  },
  play: async () => {
    const [firstInstall] = await screen.findAllByRole("button", {
      name: "Install",
    });
    await userEvent.click(firstInstall);
  },
};

export const PickExpertEmpty: Story = {
  args: PickExpert.args,
  parameters: {
    msw: {
      handlers: [
        getListExpertsMockHandler([
          expert({ id: "exp-template", is_template: true }),
        ]),
      ],
    },
  },
};

export const PickExpertLoading: Story = {
  args: PickExpert.args,
  parameters: {
    msw: { handlers: [pendingGet("*/api/experts")] },
    docs: {
      description: {
        story:
          "While the experts load, pick-expert mode has no loading state and shows “No hired experts yet.”",
      },
    },
  },
};

export const Closed: Story = {
  args: { open: false },
  parameters: {
    docs: {
      description: {
        story: "Closed: nothing is rendered and nothing is fetched.",
      },
    },
  },
};
