"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";
import NextLink from "next/link";
import { useParams } from "next/navigation";
import { Page } from "../components/Page/Page";
import { PageMenu } from "../components/PageMenu/PageMenu";
import { usePageView } from "./usePageView";

function PageViewPage() {
  const { id } = useParams<{ id: string }>();
  const view = usePageView(id);

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4 sm:p-6">
      <NextLink href="/pages" className="w-fit">
        <Text as="span" variant="small" tone="secondary">
          ← Pages
        </Text>
      </NextLink>
      {view.isLoading ? (
        <Skeleton className="h-64 w-full rounded-2xl" />
      ) : view.page && view.actions.editing ? (
        <Page
          page={view.page}
          live={view.live}
          editing={view.actions.editing}
          headerSlot={
            <PageMenu
              page={view.page}
              onRename={view.actions.rename}
              onRestore={view.actions.restore}
              onUnpin={view.actions.unpin}
              onDelete={view.actions.remove}
            />
          }
        />
      ) : (
        <Text variant="body" tone="secondary">
          This page isn&apos;t available. It may have been deleted.
        </Text>
      )}
    </main>
  );
}

export default withFeatureFlag(PageViewPage, Flag.PAGES);
