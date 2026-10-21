"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { useTallyFeedback } from "@/components/molecules/TallyPoup/useTallyFeedback";
import { useOptionalSidebar } from "@/components/ui/sidebar";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { HelpCircleIcon } from "@hugeicons/core-free-icons";

const rowClasses =
  "group relative flex w-full items-center gap-3 rounded-lg py-2 pl-3 pr-2 text-left text-sm font-normal text-neutral-700 outline-none transition-colors duration-200 ease-out hover:bg-neutral-100 focus-visible:bg-neutral-100 focus-visible:outline-none";

export function AccountMenuFeedbackRow() {
  const { isLoggedIn, isUserLoading } = useAuth();
  const feedback = useTallyFeedback(isUserLoading ? null : isLoggedIn);
  const sidebar = useOptionalSidebar();

  function handleClick() {
    if (sidebar?.isMobile) sidebar.setOpenMobile(false);
    feedback.openFeedbackForm();
  }

  return (
    <button
      type="button"
      className={rowClasses}
      onClick={handleClick}
      aria-busy={feedback.isLoading}
      data-feedback-trigger
    >
      <span
        className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-neutral-900 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100"
        aria-hidden="true"
      />
      <span className="relative z-10 flex shrink-0 items-center">
        <Icon
          icon={HelpCircleIcon}
          className="h-[18px] w-[18px] shrink-0"
          aria-hidden="true"
        />
      </span>
      <span className="relative z-10 flex-1 truncate">Give feedback</span>
    </button>
  );
}
