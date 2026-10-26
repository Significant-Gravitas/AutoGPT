import { useRef } from "react";
import { createRoot } from "react-dom/client";
import { MentionDropdown } from "@/app/(platform)/copilot/components/ChatInput/components/MentionDropdown";
import type { MentionOption } from "@/app/(platform)/copilot/components/ChatInput/useChatMentions";
import "@/app/globals.css";

const params = new URLSearchParams(location.search);
const composerAt = params.get("composer") ?? "bottom";
const fileCount = Number(params.get("files") ?? "8");
const inScroller = params.has("scroller");

const COMPOSER_POSITION: Record<string, string> = {
  top: "top-10",
  middle: "top-1/2 -mt-12",
  bottom: "bottom-10",
};

const options: MentionOption[] = Array.from({ length: fileCount }, (_, i) => ({
  kind: "file",
  file: {
    id: `file-${i + 1}`,
    name: `Report ${i + 1}.md`,
    path: `/workspace/Report ${i + 1}.md`,
    mime_type: "text/markdown",
    size_bytes: 10,
    origin: "uploaded",
    created_at: "2026-01-01T00:00:00Z",
  },
}));

function Composer({ className }: { className: string }) {
  const highlightedRef = useRef<HTMLButtonElement | null>(null);
  return (
    <form data-testid="composer" className={className}>
      <MentionDropdown
        options={options}
        showFiles
        hasIntegrations={false}
        isLoading={false}
        isError={false}
        highlightedIndex={0}
        highlightedRef={highlightedRef}
        onSelect={(option) => {
          document.body.dataset.selected =
            option.kind === "file" ? option.file.name : "";
        }}
        onHighlight={() => {}}
      />
      <textarea
        aria-label="Chat message"
        defaultValue="@"
        className="h-24 w-full rounded-3xl border border-border-base p-4"
      />
    </form>
  );
}

function Fixture() {
  if (inScroller) {
    return (
      <div
        data-testid="scroller"
        className="absolute inset-x-0 top-16 h-96 overflow-y-auto"
      >
        <div className="h-[200vh] pt-40">
          <Composer className="relative mx-2 max-w-3xl" />
        </div>
      </div>
    );
  }
  return (
    <Composer
      className={`absolute inset-x-2 mx-auto max-w-3xl ${COMPOSER_POSITION[composerAt]}`}
    />
  );
}

createRoot(document.getElementById("root")!).render(<Fixture />);
