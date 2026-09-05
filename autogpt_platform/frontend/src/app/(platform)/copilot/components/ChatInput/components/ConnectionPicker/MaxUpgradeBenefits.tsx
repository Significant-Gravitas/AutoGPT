import { AiBrain01Icon, Tick02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";

type Props = { model?: string | null };

export function MaxUpgradeBenefits({ model }: Props) {
  return (
    <>
      <p className="text-sm leading-relaxed text-zinc-600">
        Bring a more capable model to your next task, with more capacity to keep
        going.
      </p>
      <div className="mt-5 flex items-center gap-3 rounded-xl border border-purple-200/40 bg-purple-50 p-4">
        <Icon
          icon={AiBrain01Icon}
          size={22}
          className="flex-none text-purple-600"
          aria-hidden
        />
        <div className="min-w-0 flex-1">
          <p className="break-words text-sm font-medium text-zinc-900">
            {model || "Advanced models"}
          </p>
          <p className="mt-1 text-xs text-zinc-600">
            Advanced model access, included with Max
          </p>
        </div>
        <Icon
          icon={Tick02Icon}
          size={16}
          className="flex-none text-purple-600"
          aria-hidden
        />
      </div>
      <ul className="my-5 space-y-3 text-sm text-zinc-600">
        {[
          "Higher usage limits for bigger workloads",
          "More file storage and priority support",
          "Everything you already have in Pro",
        ].map((benefit) => (
          <li key={benefit} className="flex items-start gap-2.5">
            <Icon
              icon={Tick02Icon}
              size={16}
              className="mt-0.5 flex-none text-zinc-700"
              aria-hidden
            />
            {benefit}
          </li>
        ))}
      </ul>
    </>
  );
}
