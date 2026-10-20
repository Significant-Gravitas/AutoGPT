"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAdminUserLookup } from "../../useAdminUserLookup";
import { EmptyState, LookupSkeleton, NotFoundState } from "../LookupStates";
import { UserLookupBar } from "../UserLookupBar";
import { UserView } from "../UserView/UserView";

export function AdminUsersView() {
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
    <div className="flex w-full max-w-[1120px] flex-col gap-6">
      <div className="flex flex-col">
        <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
          Users
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-zinc-700">
          Look up an account by ID or email to inspect it, limit it, or act as
          the user.
        </Text>
      </div>

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
