import { ArrowUp02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  CARD_ICON_BUTTON_CLASS,
  CARD_SEND_BUTTON_CLASS,
} from "../ChatInput/helpers";

interface Props {
  hasSession: boolean;
}

// What the chat looks like before it can load, in the server's HTML too. Its
// spacing and composer classes copy ChatContainer's, EmptySession's and
// ChatInput's, so the composer stays where it is when the chat takes over.
export function ChatFrame({ hasSession }: Props) {
  return (
    <div
      data-testid="chat-frame"
      aria-busy="true"
      className="flex h-full w-full min-w-0 flex-col overflow-hidden bg-canvas"
    >
      <span role="status" className="sr-only">
        Loading your chat…
      </span>
      {hasSession ? <OpenChatFrame /> : <NewChatFrame />}
    </div>
  );
}

function NewChatFrame() {
  return (
    <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
      <div className="relative flex h-full flex-1 items-start justify-center overflow-hidden px-0 py-5 md:px-6 md:py-10">
        <div className="mx-auto w-full max-w-[52rem] pt-6">
          <div aria-hidden className="mb-1 pt-8 md:pt-0">
            <Text variant="h4" as="div">
              &nbsp;
            </Text>
          </div>
          {/* The height of EmptySession's default intro line, which wraps on a phone. */}
          <div aria-hidden className="mb-8 min-h-10 max-md:min-h-[4.75rem]" />
          <div className="mx-auto mb-6 w-full max-w-[42rem]">
            <ComposerOutline stacked />
          </div>
        </div>
      </div>
    </div>
  );
}

function OpenChatFrame() {
  return (
    <div className="flex h-full min-h-0 w-full flex-col md:px-2 lg:px-0">
      <div className="flex h-full min-h-0 w-full flex-col bg-surface">
        <div className="min-h-0 flex-1" />
        {/* oxlint-disable-next-line design-system/no-arbitrary-values -- keeps the composer above the iOS home indicator, as ChatComposerFrame does */}
        <div className="mx-auto w-full max-w-3xl px-3 pt-2 pb-[max(1.5rem,env(safe-area-inset-bottom))]">
          <ComposerOutline stacked={false} />
        </div>
      </div>
    </div>
  );
}

function ComposerOutline({ stacked }: { stacked: boolean }) {
  return (
    <div
      aria-hidden
      className={
        stacked
          ? // oxlint-disable-next-line design-system/no-arbitrary-values -- the card composer's shallow two-layer drop; no shadow token has it
            "relative z-10 flex w-full flex-col items-center gap-3 overflow-hidden rounded-3xl border border-border-base bg-surface px-3.5 pt-3 pb-3.5 shadow-[0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02)]"
          : // oxlint-disable-next-line design-system/no-arbitrary-values -- the pill composer's 2rem radius and two-layer lift; no radius or shadow token has them
            "relative z-10 flex w-full flex-col items-center overflow-hidden rounded-[2rem] border border-border-base bg-surface shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_20px_rgba(0,0,0,0.08)] max-md:rounded-3xl"
      }
    >
      {/* The text row: above the buttons in the card, and in the pill only on
          a phone, where ChatInput wraps it onto a row of its own. */}
      {stacked ? (
        <div className="min-h-[3.75rem] w-full max-md:min-h-[3.25rem]" />
      ) : (
        <div className="hidden min-h-[3.125rem] w-full max-md:block" />
      )}
      <div
        className={cn(
          "flex w-full items-center",
          stacked
            ? "gap-1.5"
            : "min-h-[3.125rem] gap-1 px-1.5 py-1 max-md:min-h-12",
        )}
      >
        <span
          className={cn(
            CARD_ICON_BUTTON_CLASS,
            "inline-flex items-center justify-center",
          )}
        >
          <Icon icon={PlusSignIcon} className="h-4 w-4" />
        </span>
        <span className="flex-1" />
        <span
          className={cn(
            CARD_SEND_BUTTON_CLASS,
            "inline-flex items-center justify-center bg-foreground-base/[0.06] text-foreground-subtle",
          )}
        >
          <Icon icon={ArrowUp02Icon} className="size-4" />
        </span>
      </div>
    </div>
  );
}
