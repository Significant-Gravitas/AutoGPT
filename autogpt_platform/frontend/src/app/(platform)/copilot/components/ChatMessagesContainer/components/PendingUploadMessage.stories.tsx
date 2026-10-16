import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { PendingUploadAttachment } from "../../../copilotStreamStore";
import { PendingUploadMessage } from "./PendingUploadMessage";

const photo: PendingUploadAttachment = {
  id: "photo",
  name: "photo.png",
  mediaType: "image/png",
  sizeBytes: 2_400_000,
  status: "uploading",
};

const report: PendingUploadAttachment = {
  id: "report",
  name: "Quarterly revenue report for the operations team - final draft v12 (reviewed).pdf",
  mediaType: "application/pdf",
  sizeBytes: 12_800_000,
  status: "uploading",
};

const notes: PendingUploadAttachment = {
  id: "notes",
  name: "notes.md",
  mediaType: "text/markdown",
  status: "uploaded",
};

const meta: Meta<typeof PendingUploadMessage> = {
  title: "Copilot/PendingUploadMessage",
  component: PendingUploadMessage,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  parameters: {
    docs: {
      description: {
        component:
          "The user's message while its files upload. Each file shows its own status; a failed file holds the message until the user retries or removes it.",
      },
    },
  },
};
export default meta;

type Story = StoryObj<typeof PendingUploadMessage>;

export const Uploading: Story = {
  args: {
    pendingSend: {
      id: "uploading",
      text: "what is this",
      attachments: [photo, { ...report }, notes],
    },
  },
};

export const OneDoneOneUploading: Story = {
  args: {
    pendingSend: {
      id: "mixed",
      text: "compare these",
      attachments: [{ ...photo, status: "uploaded" }, report],
    },
  },
};

export const OneFailed: Story = {
  args: {
    pendingSend: {
      id: "one-failed",
      text: "compare these",
      attachments: [
        { ...photo, status: "uploaded" },
        { ...report, status: "failed" },
      ],
    },
  },
};

export const FailedWhileAnotherUploads: Story = {
  args: {
    pendingSend: {
      id: "failed-uploading",
      text: "compare these",
      attachments: [{ ...photo, status: "failed" }, report],
    },
  },
};

export const SeveralFailed: Story = {
  args: {
    pendingSend: {
      id: "several-failed",
      text: "",
      attachments: [
        { ...photo, status: "failed" },
        { ...report, status: "failed" },
        notes,
      ],
    },
  },
};

export const CompactFailed: Story = {
  args: {
    isCompact: true,
    pendingSend: {
      id: "compact-failed",
      text: "what is this",
      attachments: [{ ...photo, status: "failed" }],
    },
  },
};

export const LongUnbrokenNameFailed: Story = {
  args: {
    pendingSend: {
      id: "long-name-failed",
      text: "what is this",
      attachments: [
        {
          ...report,
          name: `${"Quarterly_revenue_report_operations_team_".repeat(5)}final.pdf`,
          status: "failed",
        },
      ],
    },
  },
};
