import { useCreateRaisedExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import {
  failedAttachmentMessage,
  toRaiseAttachments,
} from "./components/KitStep/helpers";
import {
  clearDraft,
  getExpertLimitCode,
  type RaiseDraft,
  type RaiseKit,
} from "./helpers";

export function useRaiseSubmission() {
  const router = useRouter();
  const { mutateAsync: createRaisedExpert, isPending } =
    useCreateRaisedExpert();
  // The latch survives re-renders so a double click cannot fire two POSTs
  // before `isPending` has flipped; it is released again on failure so the
  // user can retry.
  const submitLatch = useRef(false);
  const [isLocked, setIsLocked] = useState(false);

  async function finish(draft: RaiseDraft, kit: RaiseKit) {
    if (submitLatch.current) return;
    submitLatch.current = true;
    setIsLocked(true);
    try {
      const response = await createRaisedExpert({
        data: {
          name: draft.name,
          role: draft.role,
          color: draft.color,
          avatar_url: draft.avatarUrl || null,
          about: draft.about || null,
          voice_preferences: draft.voicePreferences || null,
          weekly_budget: kit.weeklyBudget,
          attachments: toRaiseAttachments(kit.attachments),
        },
      });
      const result = response.data as RaiseResult;
      if (result.failed_attachments?.length) {
        toast({
          title: `Raised ${draft.name || "your expert"}, but some tools didn't attach`,
          description: failedAttachmentMessage(
            result.failed_attachments,
            kit.attachments,
          ),
        });
      }
      clearDraft();
      // kickoff=1 has the expert open the thread itself: introduce who it is,
      // say what it can take on, and start or ask for its first job.
      router.push(
        `/copilot?expertId=${encodeURIComponent(result.expert.id)}&kickoff=1`,
      );
    } catch (error) {
      submitLatch.current = false;
      setIsLocked(false);
      reportFailure(error, draft.name);
    }
  }

  return { finish, isSubmitting: isPending || isLocked };
}

function reportFailure(error: unknown, name: string) {
  if (error instanceof ApiError && error.status === 409) {
    if (getExpertLimitCode(error.response) === "raised_expert_lifetime_limit") {
      toast({
        title: "Expert creation limit reached",
        description:
          "This account has reached its lifetime raised-expert limit. Contact support if you need more capacity.",
        variant: "destructive",
      });
      return;
    }
    toast({
      title: "Your team is full",
      description:
        "You've reached the limit of active experts. Archive one from your team page to raise another.",
      variant: "destructive",
    });
    return;
  }
  toast({
    title: `Couldn't raise ${name || "your expert"}`,
    description: "Something went wrong. Please try again.",
    variant: "destructive",
  });
}
