import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  BookOpenIcon,
  ChatCircleIcon,
  FileIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";
import { useState } from "react";
import { SearchCommandModal } from "./SearchCommandModal";
import type { SearchCommandBucket } from "./helpers";

const FIXTURE_BUCKETS: SearchCommandBucket[] = [
  {
    key: "chats",
    label: "Chats",
    items: [
      {
        id: "chat-1",
        title: "Debugging the YouTube agent failures",
        icon: ChatCircleIcon,
      },
      {
        id: "chat-2",
        title: "Planning the launch announcement",
        icon: ChatCircleIcon,
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
        icon: BookOpenIcon,
      },
      {
        id: "agent-2",
        title: "Email Triage Bot",
        subtitle: "by hackergrrl",
        icon: StorefrontIcon,
      },
      {
        id: "agent-3",
        title: "PDF Question Answerer",
        subtitle: "Ask questions about uploaded PDFs",
        icon: BookOpenIcon,
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
        icon: FileIcon,
      },
      {
        id: "file-2",
        title: "competitor-analysis.md",
        subtitle: "/research",
        icon: FileIcon,
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
    // block for the modal's ``position: fixed`` descendants. Any
    // ancestor with a ``transform`` (or ``filter`` / ``contain: paint``)
    // anchors fixed children to itself instead of the viewport, which
    // lets the full dialog render inside the docs iframe without
    // overflow or scrollbars. ``[&_.fixed]:!absolute`` also retargets
    // the backdrop's positioning so it sits inside this box.
    <div
      className="relative h-[720px] w-full overflow-hidden rounded-md bg-zinc-100 [&_.fixed]:!absolute [&_.pt-\[18vh\]]:!pt-12"
      style={{ transform: "translateZ(0)" }}
    >
      {!isOpen && (
        <button
          type="button"
          className="m-4 rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-sm"
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
