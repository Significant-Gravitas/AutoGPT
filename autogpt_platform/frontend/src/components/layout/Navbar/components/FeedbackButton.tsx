"use client";

import { useTallyFeedback } from "@/components/molecules/TallyPoup/useTallyPopup";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Chatting01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function FeedbackButton() {
  const { isLoggedIn, isUserLoading } = useAuth();
  const feedback = useTallyFeedback(isUserLoading ? null : isLoggedIn);

  if (feedback.isFormVisible) return null;

  return (
    <button
      type="button"
      className="group inline-flex overflow-hidden rounded-full active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50"
      onClick={feedback.openFeedbackForm}
      disabled={feedback.isLoading}
      aria-label="Give Feedback"
      data-feedback-trigger
    >
      <div className="rounded-full bg-gradient-to-r from-indigo-100 to-indigo-300 to-zinc-400 p-[1px]">
        <div className="flex items-center gap-1.5 whitespace-nowrap rounded-full bg-[#FAFAFA]/80 px-3 py-1.5 text-sm font-medium text-neutral-700 backdrop-blur-xl transition-colors duration-150 ease-out group-hover:bg-zinc-100/90">
          <span className="hidden xl:inline">Give Feedback</span>
          <Icon icon={Chatting01Icon} size={16} />
        </div>
      </div>
    </button>
  );
}
