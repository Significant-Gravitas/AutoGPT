import { getGetV2ListLibraryFoldersMockHandler200 } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { FavoriteAnimationProvider } from "../../context/FavoriteAnimationContext";
import type { AgentStatusInfo } from "../../types";
import { LibraryAgentCard } from "./LibraryAgentCard";

const AGENT: LibraryAgent = {
  id: "4f6b2c1e-0000-4000-8000-000000000001",
  graph_id: "9a1d3e5f-0000-4000-8000-000000000001",
  graph_version: 3,
  image_url: null,
  creator_name: "Sam Rivera",
  creator_image_url: "",
  status: "HEALTHY",
  created_at: new Date("2026-09-01T09:00:00Z"),
  updated_at: new Date("2026-10-20T09:00:00Z"),
  last_run_at: new Date("2026-10-23T08:00:00Z"),
  name: "Daily inbox digest",
  description: "Summarises yesterday's email into a short morning brief.",
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
};

const IDLE_STATUS: AgentStatusInfo = {
  status: "idle",
  health: "good",
  progress: null,
  totalRuns: 12,
  lastRunAt: "2026-10-23T08:00:00Z",
  lastError: null,
  activeExecutionID: null,
  monthlySpend: 0,
  nextScheduledRun: null,
  triggerType: null,
};

const ERROR_STATUS: AgentStatusInfo = {
  ...IDLE_STATUS,
  status: "error",
  health: "attention",
  totalRuns: 7,
  lastError:
    "The Gmail credentials expired. Reconnect your Google account and run it again.",
};

const meta = {
  title: "Library/LibraryAgentCard",
  component: LibraryAgentCard,
  parameters: {
    msw: {
      handlers: [
        getGetV2ListLibraryFoldersMockHandler200({
          folders: [],
          pagination: {
            total_items: 0,
            total_pages: 0,
            current_page: 1,
            page_size: 20,
          },
        }),
      ],
    },
  },
  decorators: [
    (Story) => (
      <BackendAPIProvider>
        <TooltipProvider>
          <FavoriteAnimationProvider>
            <div className="w-[25rem] max-w-full">
              <Story />
            </div>
          </FavoriteAnimationProvider>
        </TooltipProvider>
      </BackendAPIProvider>
    ),
  ],
  args: { agent: AGENT, statusInfo: IDLE_STATUS, draggable: false },
} satisfies Meta<typeof LibraryAgentCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

/** `status === "error"` with a `lastError`: the card is wrapped in the error
 *  tooltip, which opens on hover. */
export const WithErrorTooltip: Story = {
  args: {
    agent: { ...AGENT, name: "Gmail follow-up sender", status: "ERROR" },
    statusInfo: ERROR_STATUS,
  },
};
