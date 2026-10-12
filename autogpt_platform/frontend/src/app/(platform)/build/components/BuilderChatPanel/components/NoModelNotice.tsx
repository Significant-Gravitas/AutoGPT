import { SELF_HOSTED_MODELS_DOCS_URL } from "@/app/(platform)/copilot/helpers/reportedSendError";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Alert02Icon } from "@hugeicons/core-free-icons";

// The builder chat runs only on this server's own model, so unlike the main
// chat's notice this one has no ChatGPT-plan route to offer.
export function NoModelNotice() {
  return (
    <div className="flex flex-1 items-center justify-center px-4 py-6">
      <div
        role="status"
        aria-label="No AI connection"
        className="flex w-full items-start gap-2 rounded-2xl bg-amber-50 px-3 py-2 text-left text-sm text-amber-900"
      >
        <Icon
          icon={Alert02Icon}
          className="mt-0.5 h-4 w-4 shrink-0"
          aria-hidden
        />
        <span className="min-w-0 flex-1">
          No AI model is connected, so the builder chat can&apos;t answer yet.
          Whoever runs this server can add a model API key or a local model{" "}
          <a
            href={SELF_HOSTED_MODELS_DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium underline underline-offset-2"
          >
            (setup guide)
          </a>
          .
        </span>
      </div>
    </div>
  );
}
