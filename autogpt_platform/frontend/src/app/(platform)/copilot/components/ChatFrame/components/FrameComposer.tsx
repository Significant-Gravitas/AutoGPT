import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { PromptInputTextarea } from "@/components/organisms/PromptInput/PromptInput";
import { cn } from "@/lib/utils";
import { ArrowUp02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { useState, type FormEvent } from "react";
import {
  CARD_ICON_BUTTON_CLASS,
  CARD_SEND_BUTTON_CLASS,
} from "../../ChatInput/helpers";

interface Props {
  stacked: boolean;
  placeholder: string;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

// ChatInput's composer as plain markup, its classes copied so the box does not
// move or change when the chat takes over: a working text box, and a send that
// waits for the chat. Plain elements, because the frame is in the server's
// HTML and the box must work before any of the chat's code has loaded.
export function FrameComposer({ stacked, placeholder, onSubmit }: Props) {
  const [isMultiline, setIsMultiline] = useState(false);
  return (
    <form
      onSubmit={onSubmit}
      className={
        stacked
          ? "relative w-full flex-1 max-md:[&_[data-slot=input-group-control]]:min-h-16 [&_textarea]:min-h-[4.5rem]"
          : "relative flex-1"
      }
    >
      <div
        className={
          stacked
            ? // oxlint-disable-next-line design-system/no-arbitrary-values -- the card composer's shallow two-layer drop; no shadow token has it
              "relative z-10 flex w-full min-w-0 flex-col items-center gap-3 overflow-hidden rounded-3xl border border-border-base bg-surface px-3.5 pt-3 pb-3.5 shadow-[0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02)] has-[[data-slot=input-group-control]:focus-visible]:border-border-strong"
            : // oxlint-disable-next-line design-system/no-arbitrary-values -- the pill composer's 2rem radius and two-layer lift; no radius or shadow token has them
              "relative z-10 flex w-full min-w-0 flex-col items-center overflow-hidden rounded-[2rem] border border-border-base bg-surface shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_20px_rgba(0,0,0,0.08)] has-[[data-slot=input-group-control]:focus-visible]:border-border-strong max-md:rounded-3xl"
        }
      >
        <div
          className={
            stacked
              ? "flex w-full flex-wrap items-center"
              : "flex w-full flex-wrap items-end"
          }
        >
          <div
            className={
              stacked ? "flex items-center gap-1.5" : "flex gap-1 py-1 pl-1.5"
            }
          >
            <span
              aria-hidden
              className={cn(
                CARD_ICON_BUTTON_CLASS,
                "inline-flex items-center justify-center",
              )}
            >
              <Icon icon={PlusSignIcon} className="h-4 w-4" />
            </span>
          </div>
          <div
            className={
              stacked
                ? "relative order-first w-full px-0.5 py-1"
                : isMultiline
                  ? "relative order-first w-full"
                  : "relative min-w-0 flex-1 max-md:order-first max-md:w-full max-md:basis-full"
            }
          >
            <PromptInputTextarea
              id="chat-frame-input"
              aria-label="Chat message input"
              placeholder={placeholder}
              onMultilineChange={setIsMultiline}
              className="text-left max-md:max-h-32"
            />
          </div>
          <div
            className={
              stacked
                ? "ml-auto flex items-center gap-1.5"
                : "ml-auto flex gap-1 py-1 pr-1.5"
            }
          >
            <button
              type="button"
              data-frame-send=""
              aria-label="Send message"
              onClick={(event) => event.currentTarget.form?.requestSubmit()}
              // The chat's send is disabled while the box is empty; this one
              // reads the box from CSS, so it looks right before React.
              className={
                stacked
                  ? cn(
                      CARD_SEND_BUTTON_CLASS,
                      "inline-flex shrink-0 items-center justify-center border group-has-[textarea:placeholder-shown]/frame:bg-foreground-base/[0.06] group-has-[textarea:placeholder-shown]/frame:text-foreground-subtle",
                    )
                  : "inline-flex size-[2.625rem] shrink-0 items-center justify-center rounded-full border border-action bg-action text-action-foreground group-has-[textarea:placeholder-shown]/frame:border-border-base group-has-[textarea:placeholder-shown]/frame:bg-surface-active max-md:size-10"
              }
            >
              <Icon
                icon={ArrowUp02Icon}
                className="size-4 group-data-[send-pending]/frame:hidden"
              />
              <LoadingSpinner
                size="small"
                className="hidden group-data-[send-pending]/frame:block"
              />
            </button>
          </div>
        </div>
      </div>
    </form>
  );
}
