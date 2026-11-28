import {
  getListPagesQueryKey,
  useDeletePage,
  useEditPage,
  useRemovePageSection,
  useRestorePage,
} from "@/app/api/__generated__/endpoints/pages/pages";
import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { stashCopilotPrompt } from "@/services/copilot/prompt-handoff";
import { type QueryKey, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import type { PageEditing } from "./Page";

interface Args {
  page: ResolvedPage | undefined;
  /** The query that reads this page, refreshed after every change. */
  queryKey: QueryKey;
  onDeleted?: () => void;
  onUnpinned?: () => void;
}

export function usePageActions({
  page,
  queryKey,
  onDeleted,
  onUnpinned,
}: Args) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { toast } = useToast();

  async function refresh() {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey }),
      queryClient.invalidateQueries({ queryKey: getListPagesQueryKey() }),
    ]);
  }

  const mutation = {
    onSuccess: refresh,
    onError: () =>
      toast({ title: "Couldn't change the page", variant: "destructive" }),
  };
  const edit = useEditPage({ mutation });
  const remove = useRemovePageSection({ mutation });
  const restore = useRestorePage({ mutation });
  const destroy = useDeletePage({
    mutation: { ...mutation, onSuccess: () => onDeleted?.() },
  });

  if (!page?.id) return { editing: undefined };
  const pageID = page.id;

  function move(index: number, by: -1 | 1) {
    const order = page!.sections.map((s) => s.index);
    const at = order.indexOf(index);
    const to = at + by;
    if (at < 0 || to < 0 || to >= order.length) return;
    [order[at], order[to]] = [order[to], order[at]];
    edit.mutate({ pageId: pageID, data: { order } });
  }

  function changeWithOtto(index: number, title: string | null) {
    const which = title ? `the "${title}" section` : `section ${index + 1}`;
    stashCopilotPrompt({
      prompt: `On my page "${page!.title}" (page id ${pageID}), change ${which}: `,
      autosubmit: false,
    });
    router.push(
      page!.chat_session_id
        ? `/copilot?sessionId=${encodeURIComponent(page!.chat_session_id)}`
        : "/copilot",
    );
  }

  const editing: PageEditing = {
    onMove: move,
    onRemove: (index) => remove.mutate({ pageId: pageID, index }),
    onChangeWithOtto: changeWithOtto,
  };

  return {
    editing,
    rename: (title: string) => edit.mutate({ pageId: pageID, data: { title } }),
    unpin: () =>
      edit.mutate(
        { pageId: pageID, data: { unpin: true } },
        { onSuccess: () => onUnpinned?.() },
      ),
    restore: () => restore.mutate({ pageId: pageID }),
    remove: () => destroy.mutate({ pageId: pageID }),
  };
}
