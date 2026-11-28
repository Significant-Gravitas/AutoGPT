"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import NextLink from "next/link";
import type { ReactNode } from "react";
import { Page } from "../Page/Page";
import { usePinnedPageMount } from "./usePinnedPageMount";

interface Props {
  agent: LibraryAgent;
  /** The stock view, shown when no page is pinned to the workflow. */
  children: ReactNode;
}

export function PinnedPageMount({ agent, children }: Props) {
  const mount = usePinnedPageMount(agent);

  if (!mount.enabled) return children;
  if (mount.isLoading) {
    return <Skeleton className="h-64 w-full rounded-2xl" />;
  }
  if (!mount.page?.id) return children;

  return (
    <div className="mx-auto w-full max-w-3xl p-4">
      <Page
        page={mount.page}
        live={mount.live}
        editing={mount.actions.editing}
        headerSlot={
          <div className="flex shrink-0 gap-1">
            <NextLink
              href={`/pages/${mount.page.id}`}
              className="flex items-center px-2"
            >
              <Text as="span" variant="small-medium">
                Open in Pages
              </Text>
            </NextLink>
            <Button
              variant="ghost"
              size="xs"
              onClick={() => mount.actions.unpin?.()}
            >
              Use the standard view
            </Button>
          </div>
        }
      />
    </div>
  );
}
