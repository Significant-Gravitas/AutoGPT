import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { expect, userEvent, within } from "storybook/test";
import { PromptInputSubmit } from "@/components/ai-elements/prompt-input";
import { InputGroup, InputGroupAddon } from "@/components/ui/input-group";
import { CredentialMentionEditor } from "../../CredentialMention/CredentialMentionEditor";
import { CredentialMentionMarkdown } from "../../CredentialMention/CredentialMentionMarkdown";
import { CARD_SEND_BUTTON_CLASS } from "../helpers";
import { useChatMentions } from "../useChatMentions";
import { useConnectedIntegrations } from "../useConnectedIntegrations";
import { MentionDropdown } from "./MentionDropdown";
import { mentionStoryHandlers } from "./mentionStoryFixtures";

interface Props {
  expertId?: string;
}

function AccountMentionComposer({ expertId }: Props) {
  const [value, setValue] = useState("");
  const [sent, setSent] = useState("");
  const [attachments, setAttachments] = useState<string[]>([]);
  const integrations = useConnectedIntegrations(expertId);
  const mentions = useChatMentions({
    enabled: true,
    value,
    setValue,
    integrations,
    expertId,
    includeWorkspaceFiles: true,
    addWorkspaceFile: (file) =>
      setAttachments((items) => [...items, file.name]),
    addWorkspaceFolder: (folder) =>
      setAttachments((items) => [...items, folder.name]),
  });
  return (
    <div className="w-full max-w-2xl pt-80">
      {sent && (
        <div className="mb-8 flex justify-end">
          <div
            data-testid="sent-message"
            className="max-w-full whitespace-pre-wrap rounded-3xl bg-zinc-100 px-4 py-2.5 text-base leading-8 text-zinc-900"
          >
            <CredentialMentionMarkdown>{sent}</CredentialMentionMarkdown>
          </div>
        </div>
      )}
      <form
        className="relative"
        onSubmit={(event) => {
          event.preventDefault();
          setSent(value);
          setValue("");
        }}
      >
        {mentions.isOpen && (
          <MentionDropdown
            {...mentions}
            onSelect={mentions.accept}
            onHighlight={mentions.setHighlightedIndex}
          />
        )}
        <InputGroup className="flex-col gap-3 !rounded-3xl border-zinc-200 px-3.5 pb-3.5 pt-3 shadow-sm has-[[data-slot=input-group-control]:focus-visible]:ring-0">
          {attachments.map((name) => (
            <span
              key={name}
              className="rounded-lg bg-zinc-100 px-2 py-1 text-sm"
            >
              {name}
            </span>
          ))}
          <CredentialMentionEditor
            value={value}
            onInputReady={mentions.bindInput}
            onChange={(next, input) => {
              setValue(next);
              mentions.detect(input);
            }}
            onKeyDown={(event) => {
              mentions.onKeyDown(event);
            }}
            onBlur={mentions.close}
            placeholder="Type your message..."
            className="w-full px-0.5 py-1"
          />
          <InputGroupAddon align="block-end" className="w-full justify-end p-0">
            <PromptInputSubmit
              disabled={!value.trim()}
              className={CARD_SEND_BUTTON_CLASS}
            />
          </InputGroupAddon>
        </InputGroup>
      </form>
    </div>
  );
}

const meta = {
  title: "Copilot/Account mentions",
  component: AccountMentionComposer,
  parameters: { layout: "fullscreen", msw: { handlers: mentionStoryHandlers } },
  tags: ["autodocs"],
} satisfies Meta<typeof AccountMentionComposer>;
export default meta;
type Story = StoryObj<typeof meta>;

export const AccountPicker: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.type(canvas.getByRole("textbox"), "Check my @");
    await expect(
      await canvas.findByRole("option", { name: /Work Gmail/ }),
    ).toBeVisible();
    await expect(
      await canvas.findByRole("option", { name: /Team TODOs/ }),
    ).toBeVisible();
  },
};

export const ExpertAccounts: Story = {
  args: { expertId: "expert-a" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.type(canvas.getByRole("textbox"), "Check my @");
    await expect(
      await canvas.findByRole("option", { name: /Work Gmail/ }),
    ).toBeVisible();
    await expect(
      canvas.queryByRole("option", { name: /Personal Gmail/ }),
    ).not.toBeInTheDocument();
  },
};

async function fillAccountPrompt(canvasElement: HTMLElement) {
  const canvas = within(canvasElement);
  await userEvent.type(canvas.getByRole("textbox"), "Check my @Work");
  await userEvent.click(
    await canvas.findByRole("option", { name: /Work Gmail/ }),
  );
  await userEvent.type(
    canvas.getByRole("textbox"),
    "for new TODOs, and @Personal",
  );
  await userEvent.click(
    await canvas.findByRole("option", { name: /Personal Gmail/ }),
  );
  await userEvent.type(canvas.getByRole("textbox"), "for work related emails.");
  await expect(canvas.getByRole("textbox")).toHaveTextContent("Personal Gmail");
}

export const AccountsInInput: Story = {
  play: async ({ canvasElement }) => {
    await fillAccountPrompt(canvasElement);
  },
};

export const SentMessage: Story = {
  play: async ({ canvasElement }) => {
    await fillAccountPrompt(canvasElement);
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Submit" }),
    );
    await expect(
      within(canvasElement).getByTestId("sent-message"),
    ).toHaveTextContent("Personal Gmail");
  },
};
