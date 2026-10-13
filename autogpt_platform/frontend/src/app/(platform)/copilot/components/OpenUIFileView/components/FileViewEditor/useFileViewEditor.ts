import { useContext } from "react";
import { useRouter } from "next/navigation";
import type { UIFileStateResponse } from "@/app/api/__generated__/models/uIFileStateResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { CopilotChatActionsContext } from "../../../CopilotChatActionsProvider/useCopilotChatActions";
import { chatHref, sendToChat } from "../../../../helpers/sendToChat";
import {
  fileDraftTarget,
  type DraftRemote,
} from "../../../../tools/RenderUI/draftTargets";
import { useUIDraft } from "../../../../tools/RenderUI/useUIDraft";
import { useViewSubmit } from "../../../../tools/RenderUI/useViewSubmit";
import { MISSING_ORIGIN_NOTE, resolveSubmitRoute } from "../../helpers";

interface Args {
  view: UIFileStateResponse;
  remote: DraftRemote;
  title: string;
  currentSessionId: string | null;
  readOnly: boolean;
}

export function useFileViewEditor(args: Args) {
  const { view, remote, title, currentSessionId } = args;
  const ownerId = useAuthStore((state) => state.user?.id);
  const actions = useContext(CopilotChatActionsContext);
  const router = useRouter();
  const surface = actions?.chatSurface ?? "copilot";
  const locked = args.readOnly || surface === "share";
  const draft = useUIDraft({
    namespace: `file:${view.file_id}`,
    target: fileDraftTarget({
      ownerId,
      fileId: view.file_id,
      checksum: view.checksum,
    }),
    remote,
    source: view.source,
    locked,
  });
  const route = resolveSubmitRoute({
    originSessionId: view.origin_session_id,
    currentSessionId,
    canSendHere: !!actions,
  });

  async function deliver(message: string) {
    if (route.kind === "none") return;
    if (route.kind === "here" && actions) {
      await actions.onSend(message);
      return;
    }
    if (route.kind !== "chat") return;
    await sendToChat(route.sessionId, message);
    if (route.navigate) router.push(chatHref(route.sessionId));
  }

  const submit = useViewSubmit({
    title,
    disabled: locked || !draft.writable,
    deliver: route.kind === "none" ? null : deliver,
  });

  return {
    draft,
    submit,
    locked,
    surface,
    actionsDisabled: route.kind === "none",
    note: route.kind === "none" && !locked ? MISSING_ORIGIN_NOTE : undefined,
    syncStatus: syncStatus(draft),
  };
}

function syncStatus(draft: ReturnType<typeof useUIDraft>) {
  if (!draft.ready) return "Loading…";
  if (draft.saving) return "Saving…";
  if (draft.syncError) return "Not synced";
  return "Saved";
}
