import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { http, HttpResponse } from "msw";
import { userEvent, within } from "storybook/test";
import { AssistantMessageActions } from "./AssistantMessageActions";

type StoryMessage = UIMessage<unknown, UIDataTypes, UITools>;

const REPLY_TEXT =
  "Here is the launch brief. I kept it to one page and flagged the two open pricing questions at the end.";

// A saved reply carries its database sequence at the end of its id, which is
// what makes it ratable; a reply still streaming does not have one yet.
function makeMessage(id: string): StoryMessage {
  return {
    id,
    role: "assistant",
    parts: [{ type: "text", text: REPLY_TEXT }],
  };
}

const SAVED_MESSAGE = makeMessage("story-session-seq-12");
const STREAMED_MESSAGE = makeMessage("stream-6f1c2a");

const STORY_VOICE: SpeechSynthesisVoice = {
  default: true,
  lang: "en-US",
  localService: true,
  name: "Story Voice",
  voiceURI: "story-voice",
};

// Headless browsers report no speech voices, which hides the Read aloud
// button; one stand-in voice lets the real button render.
function provideSpeechVoice() {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  window.speechSynthesis.getVoices = () => [STORY_VOICE];
}

const feedbackHandler = http.post(
  "*/api/chat/sessions/:sessionId/feedback",
  () => HttpResponse.json({}),
);

const meta: Meta<typeof AssistantMessageActions> = {
  title: "Copilot/AssistantMessageActions",
  component: AssistantMessageActions,
  parameters: {
    layout: "padded",
    msw: { handlers: [feedbackHandler] },
    docs: {
      description: {
        component:
          "Copy, rate and read-aloud actions under an assistant reply. Upvote and downvote stay disabled until the reply has been saved.",
      },
    },
  },
  args: {
    message: SAVED_MESSAGE,
    sessionID: "story-session",
  },
  decorators: [
    (Story) => {
      provideSpeechVoice();
      return (
        <TooltipProvider>
          <div className="pt-12">
            <Story />
          </div>
        </TooltipProvider>
      );
    },
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const NoFeedback: Story = {
  name: "No feedback yet",
};

export const Upvoted: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.click(
      await canvas.findByRole("button", { name: "Good response" }),
    );
  },
};

export const Downvoted: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const page = within(canvasElement.ownerDocument.body);
    await userEvent.click(
      await canvas.findByRole("button", { name: "Bad response" }),
    );
    await userEvent.type(
      await page.findByPlaceholderText(
        "Tell us what went wrong or could be improved...",
      ),
      "It missed the pricing section.",
    );
    await userEvent.click(
      await page.findByRole("button", { name: "Submit feedback" }),
    );
  },
};

export const CannotRate: Story = {
  name: "Cannot rate (reply not saved yet)",
  args: { message: STREAMED_MESSAGE },
};
