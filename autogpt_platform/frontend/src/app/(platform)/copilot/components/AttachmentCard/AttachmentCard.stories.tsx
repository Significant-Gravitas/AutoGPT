import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { AttachmentCard } from "./AttachmentCard";
import { FOLDER_MEDIA_TYPE } from "./helpers";

const IMAGE_URL = "/images/team-card-banner.jpg";
const LONG_NAME =
  "Quarterly revenue report for the operations team - final draft v12 (reviewed).pdf";

const meta: Meta<typeof AttachmentCard> = {
  title: "Copilot/AttachmentCard",
  component: AttachmentCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "One attachment, in the composer and above a sent message: images as square thumbnails, other files as a card with their type icon, name, and type · size. Pending messages add an upload status, and a failed upload offers Retry and Remove.",
      },
    },
  },
  args: { size: "message" },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Image: Story = {
  args: { name: "photo.jpg", mediaType: "image/jpeg", previewUrl: IMAGE_URL },
};

export const Pdf: Story = {
  args: {
    name: "report.pdf",
    mediaType: "application/pdf",
    sizeBytes: 1_260_000,
    onOpen: () => {},
  },
};

export const LongName: Story = {
  args: {
    name: LONG_NAME,
    mediaType: "application/pdf",
    sizeBytes: 12_800_000,
  },
};

export const Folder: Story = {
  args: {
    name: "Q3 reports",
    mediaType: FOLDER_MEDIA_TYPE,
    detail: "Folder · 12 files",
    href: "/artifacts?folder=q3",
  },
};

export const ImageUploading: Story = {
  args: {
    name: "photo.jpg",
    mediaType: "image/jpeg",
    previewUrl: IMAGE_URL,
    status: "uploading",
  },
};

export const FileUploading: Story = {
  args: {
    name: "notes.md",
    mediaType: "text/markdown",
    sizeBytes: 4_200,
    status: "uploading",
  },
};

export const Failed: Story = {
  args: {
    name: LONG_NAME,
    mediaType: "application/pdf",
    sizeBytes: 12_800_000,
    status: "failed",
    onRetry: () => {},
    onRemove: () => {},
  },
};

export const ComposerRow: Story = {
  render: () => (
    <div className="flex max-w-md flex-wrap gap-2 rounded-xl border border-zinc-200 p-3">
      <AttachmentCard
        size="composer"
        name="photo.jpg"
        mediaType="image/jpeg"
        previewUrl={IMAGE_URL}
        onRemove={() => {}}
      />
      <AttachmentCard
        size="composer"
        name={LONG_NAME}
        mediaType="application/pdf"
        sizeBytes={12_800_000}
        onRemove={() => {}}
      />
      <AttachmentCard
        size="composer"
        name="budget.xlsx"
        mediaType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        sizeBytes={48_000}
        onRemove={() => {}}
      />
      <AttachmentCard
        size="composer"
        name="main.py"
        mediaType="text/x-python"
        sizeBytes={2_100}
        onRemove={() => {}}
      />
    </div>
  ),
};
