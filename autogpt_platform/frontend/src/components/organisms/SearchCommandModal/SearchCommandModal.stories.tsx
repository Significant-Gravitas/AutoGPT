import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { SearchCommandModal } from "./SearchCommandModal";
import type { SearchCommandBucket } from "./helpers";
import {
  BookOpen01Icon,
  BubbleChatIcon,
  FileEmpty02Icon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import { createIconComponent } from "@/components/atoms/Icon/Icon";

const FIXTURE_BUCKETS: SearchCommandBucket[] = [
  {
    key: "chats",
    label: "Chats",
    items: [
      {
        id: "chat-1",
        title: "Debugging the YouTube agent failures",
        icon: createIconComponent(BubbleChatIcon),
      },
      {
        id: "chat-2",
        title: "Planning the launch announcement",
        icon: createIconComponent(BubbleChatIcon),
      },
    ],
  },
  {
    key: "agents",
    label: "Agents",
    items: [
      {
        id: "agent-1",
        title: "YouTube Video Summarizer",
        subtitle: "Summarize any YouTube video into bullet points",
        icon: createIconComponent(BookOpen01Icon),
      },
      {
        id: "agent-2",
        title: "Email Triage Bot",
        subtitle: "by hackergrrl",
        icon: createIconComponent(Store01Icon),
      },
      {
        id: "agent-3",
        title: "PDF Question Answerer",
        subtitle: "Ask questions about uploaded PDFs",
        icon: createIconComponent(BookOpen01Icon),
      },
    ],
  },
  {
    key: "files",
    label: "Files",
    items: [
      {
        id: "file-1",
        title: "Q4-roadmap.pdf",
        subtitle: "/projects/planning",
        icon: createIconComponent(FileEmpty02Icon),
      },
      {
        id: "file-2",
        title: "competitor-analysis.md",
        subtitle: "/research",
        icon: createIconComponent(FileEmpty02Icon),
      },
    ],
  },
];

interface DemoArgs {
  buckets: SearchCommandBucket[];
  isLoading?: boolean;
  isError?: boolean;
  initialQuery?: string;
  placeholder?: string;
  inputAriaLabel?: string;
  idleEmptyLabel?: string;
  searchingEmptyLabel?: string;
}

function SearchCommandModalDemo({
  buckets,
  isLoading,
  isError,
  initialQuery = "",
  placeholder,
  inputAriaLabel,
  idleEmptyLabel,
  searchingEmptyLabel,
}: DemoArgs) {
  const [isOpen, setIsOpen] = useState(true);
  const [query, setQuery] = useState(initialQuery);

  const trimmed = query.trim().toLowerCase();
  // Lightweight in-memory filter so the story behaves like a real
  // command palette as you type. Empty query shows the full fixture
  // (which acts as the "recent items" state in the real app).
  const filteredBuckets = trimmed
    ? buckets.map((bucket) => ({
        ...bucket,
        items: bucket.items.filter((item) =>
          item.title.toLowerCase().includes(trimmed),
        ),
      }))
    : buckets;

  return (
    // Storybook preview shim: a sized box that becomes the containing
    // block for the modal's ``position: fixed`` overlay. Any ancestor
    // with a ``transform`` (or ``filter`` / ``contain: paint``) anchors
    // fixed children to itself instead of the viewport, which lets the
    // full dialog render inside the docs iframe without overflow or
    // scrollbars. The overlay's ``18vh`` top padding is pulled in so the
    // dialog sits inside the box at any iframe height.
    <div
      className="relative h-[720px] w-full overflow-hidden rounded-md bg-zinc-100 [&_.command-overlay]:pt-12!"
      style={{ transform: "translateZ(0)" }}
    >
      {!isOpen && (
        <button
          type="button"
          className="m-4 rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-xs"
          onClick={() => setIsOpen(true)}
        >
          Open search
        </button>
      )}
      <SearchCommandModal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        query={query}
        onQueryChange={setQuery}
        buckets={filteredBuckets}
        isLoading={isLoading}
        isError={isError}
        placeholder={placeholder}
        inputAriaLabel={inputAriaLabel}
        idleEmptyLabel={idleEmptyLabel}
        searchingEmptyLabel={searchingEmptyLabel}
        onSelectItem={() => {
          setIsOpen(false);
        }}
      />
    </div>
  );
}

const meta: Meta<typeof SearchCommandModalDemo> = {
  title: "Organisms/SearchCommandModal",
  component: SearchCommandModalDemo,
  tags: ["autodocs"],
  parameters: {
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Generic, controlled command-palette modal with bucketed results, keyboard navigation, and slot-based empty/error/loading states. App-agnostic — see ``GlobalSearchModal`` for the wired-up container that talks to the backend.",
      },
      // Safety net so the autodocs iframe is always tall enough for
      // the sized wrapper that anchors the modal inside the preview.
      // See the comment on the wrapper ``<div>`` in
      // ``SearchCommandModalDemo`` for the positioning trick.
      story: { iframeHeight: 800 },
    },
  },
  args: {
    buckets: FIXTURE_BUCKETS,
    placeholder: "Search agents, files, chats…",
    inputAriaLabel: "Global search",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const WithResults: Story = {};

export const Loading: Story = {
  args: {
    buckets: [],
    isLoading: true,
  },
};

export const Empty: Story = {
  args: {
    buckets: [],
    idleEmptyLabel: "No recent items",
  },
};

export const NoMatches: Story = {
  args: {
    buckets: [],
    initialQuery: "zzz",
    searchingEmptyLabel: "No results found",
  },
};

export const SingleBucket: Story = {
  args: {
    buckets: [FIXTURE_BUCKETS[0]],
  },
};

export const ErrorState: Story = {
  args: {
    buckets: [],
    isError: true,
  },
};
