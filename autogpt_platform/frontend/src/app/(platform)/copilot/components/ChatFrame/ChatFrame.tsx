import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { FRAME_LOADING_TEXT, type FrameSnapshot } from "./chatFrameHandoff";
import { FrameComposer } from "./components/FrameComposer";
import { FRAME_SCRIPT } from "./frameScript";
import { useChatFrame } from "./useChatFrame";

interface Props {
  hasSession: boolean;
  /** Receives what the box held as the frame leaves, for the chat to take. */
  onUnmount?: (snapshot: FrameSnapshot | null) => void;
  className?: string;
}

// What the chat looks like while it loads, in the server's HTML too. The
// message box is a real one: typing goes there until the chat takes the box
// over, text, caret and all. Its spacing copies ChatContainer's and
// EmptySession's, so the box stays where it is when the chat takes over.
export function ChatFrame({ hasSession, onUnmount, className }: Props) {
  const { ref, isHydrating, composer } = useChatFrame(hasSession, onUnmount);
  return (
    <div
      ref={ref}
      data-testid="chat-frame"
      // A send pending is an attribute the inline script below can set before
      // React hydrates; React never renders it, and reads it from the DOM.
      suppressHydrationWarning
      className={cn(
        "group/frame flex h-full w-full min-w-0 flex-col overflow-hidden bg-canvas",
        className,
      )}
    >
      {/* markSendPending rewrites this, before hydration too. */}
      <div role="status" className="sr-only" suppressHydrationWarning>
        {FRAME_LOADING_TEXT}
      </div>
      {hasSession ? (
        <OpenChatFrame composer={<FrameComposer {...composer} />} />
      ) : (
        <NewChatFrame composer={<FrameComposer {...composer} />} />
      )}
      {isHydrating && (
        <script dangerouslySetInnerHTML={{ __html: FRAME_SCRIPT }} />
      )}
    </div>
  );
}

function NewChatFrame({ composer }: { composer: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
      <div className="relative flex h-full flex-1 items-start justify-center overflow-hidden px-0 py-5 md:px-6 md:py-10">
        <div className="mx-auto w-full max-w-[52rem] pt-6">
          <div aria-busy="true" aria-hidden className="text-center">
            <div className="mb-1 pt-8 md:pt-0">
              <Text variant="h4" as="div" className="relative">
                &nbsp;
                <Skeleton className="absolute inset-y-1 left-1/2 w-36 -translate-x-1/2 rounded-full" />
              </Text>
            </div>
            {/* The height of EmptySession's default intro line, which wraps on a phone. */}
            <div className="relative mb-8 min-h-10 max-md:min-h-[4.75rem]">
              <Skeleton className="absolute top-2 left-1/2 h-6 w-[min(32rem,90%)] -translate-x-1/2 rounded-full" />
            </div>
          </div>
          <div className="mx-auto mb-6 w-full max-w-[42rem] text-left">
            {composer}
          </div>
        </div>
      </div>
    </div>
  );
}

function OpenChatFrame({ composer }: { composer: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col md:px-2 lg:px-0">
      <div className="flex h-full min-h-0 w-full flex-col bg-surface">
        <div
          aria-busy="true"
          className="flex min-h-0 flex-1 items-center justify-center text-foreground-secondary"
        >
          <LoadingSpinner />
        </div>
        {/* oxlint-disable-next-line design-system/no-arbitrary-values -- keeps the composer above the iOS home indicator, as ChatComposerFrame does */}
        <div className="mx-auto w-full max-w-3xl px-3 pt-2 pb-[max(1.5rem,env(safe-area-inset-bottom))]">
          {composer}
        </div>
      </div>
    </div>
  );
}
