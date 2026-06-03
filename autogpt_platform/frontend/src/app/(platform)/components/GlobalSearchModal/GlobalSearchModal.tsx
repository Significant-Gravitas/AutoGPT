import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { SearchCommandModal } from "@/components/organisms/SearchCommandModal/SearchCommandModal";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ACTIONS_BUCKET_KEY, COPY_USER_ID_ACTION } from "./actions";
import { NAV_BUCKET_KEY, getNavigationHref } from "./navigation";
import { useGlobalSearch } from "./useGlobalSearch";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelectItem: (item: SearchResultItem) => void;
}

export function GlobalSearchModal({ isOpen, onClose, onSelectItem }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const { toast } = useToast();
  const { user } = useSupabase();
  const { query, setQuery, buckets, itemsById, isFetching, isError } =
    useGlobalSearch(isOpen);
  // Id of the row whose select is in-flight. The modal stays open with a
  // trailing spinner on that row while the work runs — a navigation
  // routing to a (possibly slow) page, or an action awaiting a side
  // effect like a clipboard write.
  const [busyItemId, setBusyItemId] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) setBusyItemId(null);
  }, [isOpen]);

  // A navigation row keeps the spinner up until the route actually
  // changes (router.push resolves async). Clearing on the resolved
  // pathname is the real end of the action — don't rely on the tree
  // unmounting, so the row never stays stuck if this modal ever lives
  // in a layout that persists across pages.
  useEffect(() => {
    setBusyItemId(null);
  }, [pathname]);

  async function runAction(id: string) {
    if (id === COPY_USER_ID_ACTION) {
      if (!user?.id) {
        toast({ title: "You're not signed in", variant: "destructive" });
        return;
      }
      try {
        await navigator.clipboard.writeText(user.id);
        toast({ title: "User ID copied to clipboard" });
      } catch {
        toast({ title: "Couldn't copy User ID", variant: "destructive" });
      }
    }
  }

  function handleSelect(id: string, bucketKey: string) {
    if (busyItemId) return;
    if (bucketKey === NAV_BUCKET_KEY) {
      const href = getNavigationHref(id);
      if (!href) return;
      // No close: the destination page mounting unmounts this tree.
      setBusyItemId(id);
      router.push(href);
      return;
    }
    if (bucketKey === ACTIONS_BUCKET_KEY) {
      setBusyItemId(id);
      void runAction(id).finally(onClose);
      return;
    }
    const apiItem = itemsById.get(id);
    if (!apiItem) return;
    onSelectItem(apiItem);
    onClose();
  }

  return (
    <SearchCommandModal
      isOpen={isOpen}
      onClose={onClose}
      query={query}
      onQueryChange={setQuery}
      buckets={buckets}
      isLoading={isFetching}
      isError={isError}
      placeholder="Search agents, files, chats..."
      inputAriaLabel="Global search"
      idleEmptyLabel="No recent items"
      searchingEmptyLabel="No results found"
      loadingItemId={busyItemId ?? undefined}
      onSelectItem={(item, bucketKey) => handleSelect(item.id, bucketKey)}
    />
  );
}
