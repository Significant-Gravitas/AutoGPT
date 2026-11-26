import type { Meta, StoryObj } from "@storybook/nextjs";

import {
  DroppedSendError,
  UnconfirmedSendError,
} from "../../../../stream/turnRuntime";
import { ChatErrorBanner } from "./ChatErrorBanner";

const meta: Meta<typeof ChatErrorBanner> = {
  title: "Copilot/ChatErrorBanner",
  component: ChatErrorBanner,
  parameters: {
    docs: {
      description: {
        component:
          "The error under the chat: what happened, and the one action that can work next.",
      },
    },
  },
};
export default meta;

type Story = StoryObj<typeof ChatErrorBanner>;

/** A send no check could confirm, which the page can send again. */
export const UnconfirmedSend: Story = {
  args: {
    error: new UnconfirmedSendError("local:1"),
    onResend: () => {},
  },
};

/** The same where nothing can send it again, as while a turn runs. */
export const UnconfirmedSendWhileBusy: Story = {
  args: {
    error: new UnconfirmedSendError("local:1"),
  },
};

/** A send the network dropped, as when offline at send; it can go again too. */
export const DroppedSend: Story = {
  args: {
    error: new DroppedSendError("local:1", new TypeError("Failed to fetch")),
    onResend: () => {},
  },
};
