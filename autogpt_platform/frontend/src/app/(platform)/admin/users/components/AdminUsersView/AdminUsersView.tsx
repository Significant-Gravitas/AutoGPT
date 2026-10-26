"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import {
  SecurityBlockIcon,
  UserSearch01Icon,
} from "@hugeicons/core-free-icons";
import { useQueryState } from "nuqs";
import { BLOCKS_TAB, TAB_QUERY_PARAM } from "../../helpers";
import { useAdminUserLookup } from "../../useAdminUserLookup";
import { BlocksTab } from "../BlocksTab/BlocksTab";
import { useActiveBlockCount } from "../BlocksTab/useBlocksTab";
import { EmptyState, LookupSkeleton, NotFoundState } from "../LookupStates";
import { UserLookupBar } from "../UserLookupBar";
import { UserView } from "../UserView/UserView";

const LOOKUP_TAB = "lookup";

export function AdminUsersView() {
  const [tab, setTab] = useQueryState(TAB_QUERY_PARAM);
  const activeTab = tab === BLOCKS_TAB ? BLOCKS_TAB : LOOKUP_TAB;
  const blockCount = useActiveBlockCount();

  return (
    <div className="flex w-full max-w-[1120px] flex-col gap-6">
      <Text variant="h4" as="h1" className="leading-7 text-textBlack">
        Users
      </Text>
      <TabsLine
        value={activeTab}
        onValueChange={(value) =>
          void setTab(value === BLOCKS_TAB ? BLOCKS_TAB : null)
        }
      >
        <TabsLineList flush indicatorClassName="bg-action">
          <TabsLineTrigger value={LOOKUP_TAB} icon={UserSearch01Icon}>
            Lookup
          </TabsLineTrigger>
          <TabsLineTrigger value={BLOCKS_TAB} icon={SecurityBlockIcon}>
            <span className="flex items-center gap-1.5">
              Blocks
              {blockCount ? (
                <Badge variant="info" size="small">
                  {blockCount.toLocaleString("en-GB")}
                </Badge>
              ) : null}
            </span>
          </TabsLineTrigger>
        </TabsLineList>
        <TabsLineContent value={LOOKUP_TAB} className="mt-4">
          <LookupTab />
        </TabsLineContent>
        <TabsLineContent value={BLOCKS_TAB} className="mt-4">
          <div className="flex flex-col gap-6">
            <Text variant="body" className="max-w-[600px] text-foreground-body">
              IP addresses refused at sign-up and sign-in, and email domains
              refused at sign-up.
            </Text>
            <BlocksTab />
          </div>
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}

function LookupTab() {
  const {
    rawQuery,
    query,
    user,
    isLoading,
    isNotFound,
    error,
    search,
    refetch,
  } = useAdminUserLookup();

  return (
    <div className="flex flex-col gap-6">
      <Text variant="body" className="max-w-[600px] text-foreground-body">
        Look up an account by ID or email to inspect it, limit it, or act as the
        user.
      </Text>

      <UserLookupBar
        key={rawQuery}
        initialValue={rawQuery}
        isLoading={isLoading}
        showHint={query === null}
        onSearch={search}
      />

      {query === null ? (
        <EmptyState />
      ) : isLoading ? (
        <LookupSkeleton />
      ) : error ? (
        <ErrorCard
          context="user lookup"
          responseError={{ message: error.message }}
          onRetry={() => {
            void refetch();
          }}
        />
      ) : isNotFound || !user ? (
        <NotFoundState query={rawQuery.trim()} />
      ) : (
        <UserView user={user} />
      )}
    </div>
  );
}
