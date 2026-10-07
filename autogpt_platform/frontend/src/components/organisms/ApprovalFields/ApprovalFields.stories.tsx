import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { expect, userEvent, within } from "storybook/test";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { MANAGED_IDENTITIES } from "@/components/molecules/ExpertAvatar/helpers";
import { ApprovalFields } from "./ApprovalFields";
import type { Reference } from "./helpers";

const LONG_BODY = [
  "Hi Dana,",
  "",
  "Thanks for the call this morning. As promised, here is a summary of the Q3 numbers we went through together, plus the follow-ups we agreed on.",
  "Revenue landed 12% above plan, driven mostly by the enterprise tier. Churn held flat at 2.1%.",
  "I will send the full deck by Friday once finance signs off on the final figures.",
  "",
  "Best,",
  "Sam",
].join("\n");

function reference(overrides: Partial<Reference> = {}): Reference {
  return {
    key: "folder_id",
    entity: "library_folder",
    id: "f-q3",
    name: "Q3 reports",
    href: "/library?folder=f-q3",
    kind: "Library folder",
    description: null,
    meta: [
      { text: "4 agents", cron: null, label: null, at: null },
      { text: "1 subfolder", cron: null, label: null, at: null },
      { text: "In Finance", cron: null, label: null, at: null },
    ],
    avatarURL: null,
    avatarColor: null,
    skills: [],
    summary: "4 agents · 1 subfolder · In Finance",
    ...overrides,
  };
}

const meta = {
  title: "Organisms/ApprovalFields",
  component: ApprovalFields,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="w-lg">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "The arguments of a held tool or block call, as a label/value list on an approval card. Each value is drawn by its kind: short text, secret, code, long (clamped) text, list, flat object or JSON. Id arguments the server resolved show as named, linked references with a hover card. At most six fields show before a “Show N more” link; it renders nothing when no field has a value.",
      },
    },
  },
  args: {
    fields: [
      { key: "to", label: "To" },
      { key: "subject", label: "Subject" },
      { key: "body", label: "Body" },
    ],
    values: {
      to: ["dana@example.com"],
      subject: "Q3 numbers recap",
      body: "Thanks for the call. Full deck by Friday.",
    },
  },
} satisfies Meta<typeof ApprovalFields>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const AllValueKinds: Story = {
  args: {
    fields: [
      { key: "title", label: "Title" },
      { key: "api_key", label: "API key" },
      { key: "command", label: "Command" },
      { key: "body", label: "Body" },
      { key: "recipients", label: "Recipients" },
      { key: "options", label: "Options" },
    ],
    values: {
      title: "Weekly sales sync",
      api_key: "[redacted]",
      command: "git fetch origin\ngit rebase origin/main\npnpm test",
      body: LONG_BODY,
      recipients: ["dana@example.com", "sam@example.com"],
      options: { notify: true, retries: 3, labels: ["finance", "q3"] },
    },
  },
};

export const SecretValue: Story = {
  args: {
    fields: [
      { key: "provider", label: "Provider" },
      { key: "api_key", label: "API key" },
    ],
    values: { provider: "OpenAI", api_key: "[redacted]" },
  },
};

export const CodeValue: Story = {
  args: {
    fields: [{ key: "sql", label: "Query" }],
    values: {
      sql: "SELECT id, email\nFROM users\nWHERE created_at > '2026-01-01'\nORDER BY created_at DESC\nLIMIT 50;",
    },
  },
};

export const LongTextClamped: Story = {
  args: {
    fields: [{ key: "body", label: "Body" }],
    values: { body: LONG_BODY },
  },
};

export const LongTextExpanded: Story = {
  args: LongTextClamped.args,
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: /Show all/ }),
    );
  },
};

export const LongList: Story = {
  args: {
    fields: [{ key: "to", label: "To" }],
    values: {
      to: [
        "dana@example.com",
        "sam@example.com",
        "lee@example.com",
        "ana@example.com",
        "kim@example.com",
        "max@example.com",
        "zoe@example.com",
      ],
    },
  },
};

export const FlatObject: Story = {
  args: {
    fields: [{ key: "settings", label: "Settings" }],
    values: {
      settings: {
        timezone: "Europe/London",
        notify: false,
        token: "[redacted]",
        channels: ["email", "slack"],
      },
    },
  },
};

export const NestedJson: Story = {
  args: {
    fields: [{ key: "payload", label: "Payload" }],
    values: {
      payload: {
        customer: { id: "c-42", name: "Acme" },
        items: [{ sku: "A-1", qty: 2 }],
      },
    },
  },
};

export const NestedJsonOpen: Story = {
  args: NestedJson.args,
  play: async ({ canvasElement }) => {
    await userEvent.click(within(canvasElement).getByText("View as JSON"));
  },
};

export const ClippedByServer: Story = {
  args: {
    fields: [
      { key: "subject", label: "Subject" },
      { key: "to", label: "To" },
    ],
    values: {
      subject: "Q3 numbers recap",
      to: ["dana@example.com", "sam@example.com"],
    },
    clipped: ["to"],
  },
  parameters: {
    docs: {
      description: {
        story:
          "A value the server shortened is marked “(shortened)”; the approval still binds the whole value.",
      },
    },
  },
};

export const ManyFields: Story = {
  args: {
    fields: [
      { key: "name", label: "Name" },
      { key: "email", label: "Email" },
      { key: "company", label: "Company" },
      { key: "role", label: "Role" },
      { key: "city", label: "City" },
      { key: "country", label: "Country" },
      { key: "phone", label: "Phone" },
      { key: "source", label: "Source" },
    ],
    values: {
      name: "Dana Whitfield",
      email: "dana@example.com",
      company: "Acme",
      role: "Head of Finance",
      city: "London",
      country: "United Kingdom",
      phone: "+44 20 7946 0000",
      source: "Webinar",
    },
  },
};

export const ManyFieldsExpanded: Story = {
  args: ManyFields.args,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.click(canvas.getByRole("button", { name: "Show 2 more" }));
    await expect(canvas.getByText("Webinar")).toBeInTheDocument();
  },
};

export const UnlabelledKeysAreHumanized: Story = {
  args: {
    fields: [],
    values: {
      search_query: "quarterly report",
      maxResults: 10,
      is_draft: true,
    },
  },
};

export const HiddenKeys: Story = {
  args: {
    fields: [
      { key: "folder_name", label: "Folder" },
      { key: "parent", label: "Parent" },
    ],
    values: { folder_name: "Q3 reports", parent: "Finance" },
    hiddenKeys: ["folder_name"],
  },
  parameters: {
    docs: {
      description: {
        story: "Keys the card headline already names are left out.",
      },
    },
  },
};

export const IdsHidden: Story = {
  args: {
    fields: [
      { key: "agent_id", label: "Agent" },
      { key: "note", label: "Note" },
    ],
    values: { agent_id: "a-7f3c", note: "Re-run with the new prompt" },
  },
  parameters: {
    docs: {
      description: {
        story:
          "Raw ids are dropped when other arguments already tell the call apart.",
      },
    },
  },
};

export const IdsWhenAlone: Story = {
  args: {
    fields: [{ key: "agent_id", label: "Agent" }],
    values: { agent_id: "a-7f3c" },
    idsWhenAlone: true,
  },
};

export const LinkedReference: Story = {
  args: {
    fields: [{ key: "folder_id", label: "Folder" }],
    values: { folder_id: "f-q3" },
    references: [reference()],
    referenceTotals: { folder_id: 1 },
  },
};

export const ReferenceHoverCard: Story = {
  args: LinkedReference.args,
  play: async ({ canvasElement }) => {
    await userEvent.hover(
      within(canvasElement).getByRole("link", { name: "Q3 reports" }),
    );
  },
};

export const ExpertReferenceWithoutPage: Story = {
  args: {
    fields: [{ key: "expert_id", label: "Expert" }],
    values: { expert_id: "e-maria" },
    references: [
      reference({
        key: "expert_id",
        entity: "expert",
        id: "e-maria",
        name: "Maria",
        href: null,
        kind: "Expert",
        description: "Keeps the books tidy and chases late invoices.",
        meta: [],
        avatarURL: MANAGED_IDENTITIES[0].url,
        skills: ["Invoicing", "Reconciliation", "Forecasting", "Payroll"],
        summary: null,
      }),
    ],
  },
  parameters: {
    docs: {
      description: {
        story:
          "A reference with no page of its own is named without a link, but stays focusable so the keyboard can open its card.",
      },
    },
  },
};

export const SeveralReferences: Story = {
  args: {
    fields: [{ key: "agent_ids", label: "Agents" }],
    values: { agent_ids: ["a-1", "a-2", "a-3", "a-4", "a-5"] },
    references: [
      reference({
        key: "agent_ids",
        entity: "library_agent",
        id: "a-1",
        name: "Lead finder",
        href: "/library/agents/a-1",
        kind: null,
        summary: null,
        meta: [],
      }),
      reference({
        key: "agent_ids",
        entity: "library_agent",
        id: "a-2",
        name: "Invoice chaser",
        href: "/library/agents/a-2",
        kind: null,
        summary: null,
        meta: [],
      }),
    ],
    referenceTotals: { agent_ids: 5 },
  },
};

export const UnresolvedReference: Story = {
  args: {
    fields: [{ key: "folder_id", label: "Folder" }],
    values: { folder_id: "f-missing" },
    references: [
      reference({
        id: "f-missing",
        name: null,
        href: null,
        kind: null,
        summary: null,
        meta: [],
      }),
    ],
  },
  parameters: {
    docs: {
      description: {
        story: "An id the server could not name shows as the raw id.",
      },
    },
  },
};

export const Empty: Story = {
  args: {
    fields: [{ key: "note", label: "Note" }],
    values: { note: "", archived: false, tags: [] },
  },
  parameters: {
    docs: {
      description: {
        story:
          "Empty strings, `false`, empty lists and empty objects count as no value; with nothing left the component renders nothing.",
      },
    },
  },
};
