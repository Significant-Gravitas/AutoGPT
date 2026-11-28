"use client";

import { useListPages } from "@/app/api/__generated__/endpoints/pages/pages";
import type { PageListItem } from "@/app/api/__generated__/models/pageListItem";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { stashCopilotPrompt } from "@/services/copilot/prompt-handoff";
import { CollectionsList } from "@/app/(platform)/collections/components/CollectionsList/CollectionsList";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";
import { formatDate } from "@/lib/utils/time";
import NextLink from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

const ASK_PROMPT = "Make me a page that ";

function PagesListPage() {
  const collectionsEnabled = Boolean(useGetFlag(Flag.COLLECTIONS));
  const tab = useSearchParams().get("tab");
  const router = useRouter();
  if (!collectionsEnabled) return <PagesTab router={router} />;
  return (
    <TabsLine
      value={tab === "collections" ? "collections" : "pages"}
      onValueChange={(value) =>
        router.replace(
          value === "collections" ? "/pages?tab=collections" : "/pages",
        )
      }
    >
      <div className="mx-auto w-full max-w-3xl px-4 pt-4 sm:px-6">
        <TabsLineList>
          <TabsLineTrigger value="pages">Pages</TabsLineTrigger>
          <TabsLineTrigger value="collections">Collections</TabsLineTrigger>
        </TabsLineList>
      </div>
      <TabsLineContent value="pages">
        <PagesTab router={router} />
      </TabsLineContent>
      <TabsLineContent value="collections">
        <main className="mx-auto flex w-full max-w-3xl flex-col gap-6 p-4 sm:p-6">
          <Text variant="h2" as="h1">
            Collections
          </Text>
          <CollectionsList />
        </main>
      </TabsLineContent>
    </TabsLine>
  );
}

function PagesTab({ router }: { router: ReturnType<typeof useRouter> }) {
  const query = useListPages({ query: { select: okData } });
  const pages = query.data?.pages ?? [];

  function askOtto() {
    stashCopilotPrompt({ prompt: ASK_PROMPT, autosubmit: false });
    router.push("/copilot");
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-6 p-4 sm:p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <Text variant="h2" as="h1">
          Pages
        </Text>
        <Button size="small" onClick={askOtto}>
          Ask Otto for a page
        </Button>
      </header>
      {query.isLoading ? (
        <Skeleton className="h-40 w-full rounded-2xl" />
      ) : pages.length ? (
        <ul className="flex flex-col gap-2">
          {pages.map((page) => (
            <li key={page.id}>
              <PageRow page={page} />
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex flex-col items-start gap-3 rounded-2xl bg-surface p-6 smooth-shadow-ring-sm">
          <Text variant="body-medium">No pages yet</Text>
          <Text variant="body" tone="secondary">
            Ask Otto for a page that brings your workflows and experts together:
            a form to run one, its latest result, what an expert has done, and
            the approvals waiting on you.
          </Text>
        </div>
      )}
    </main>
  );
}

function PageRow({ page }: { page: PageListItem }) {
  const parts = [
    count(page.workflow_count, "workflow"),
    count(page.expert_count, "expert"),
  ].filter(Boolean);
  return (
    <NextLink
      href={`/pages/${page.id}`}
      className="flex items-center justify-between gap-3 rounded-2xl bg-surface p-4 smooth-shadow-ring-sm hover:bg-surface-muted"
    >
      <span className="flex min-w-0 flex-col gap-0.5">
        <Text variant="body-medium" className="truncate">
          {page.title}
        </Text>
        <Text variant="small" tone="secondary">
          {[...parts, page.pinned_library_agent_id ? "pinned" : null]
            .filter(Boolean)
            .join(" · ")}
        </Text>
      </span>
      <Text variant="small" tone="secondary" className="shrink-0">
        {formatDate(new Date(page.updated_at))}
      </Text>
    </NextLink>
  );
}

function count(n: number, noun: string) {
  if (!n) return null;
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

export default withFeatureFlag(PagesListPage, Flag.PAGES);
