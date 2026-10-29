import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { tourStartingPoint } from "@/lib/tour-targets";
import { cn } from "@/lib/utils";
import { Cancel01Icon, Loading03Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import {
  getChatHref,
  getMentionedProviders,
  getPromptIcon,
  getTaskState,
} from "./helpers";

interface Props {
  task: SuggestedTask;
  isSending: boolean;
  isBusy: boolean;
  canHide: boolean;
  onSend: (task: SuggestedTask) => void;
  onHide: (task: SuggestedTask) => void;
}

const CARD_CLASSES =
  "flex w-full flex-col gap-3 rounded-2xl border border-border-base bg-surface p-4 text-left transition-[border-color,box-shadow] duration-150";

export function SuggestedTaskCard({
  task,
  isSending,
  isBusy,
  canHide,
  onSend,
  onHide,
}: Props) {
  const isSent = getTaskState(task) === "sent";

  return (
    <>
      {isSent && task.session_id ? (
        <NextLink
          href={getChatHref(task.session_id)}
          className={cn(CARD_CLASSES, "hover:border-border-strong")}
        >
          <CardTop task={task} muted />
          <Text variant="body-medium" tone="muted" className="text-pretty">
            {task.title}
          </Text>
          <Text
            variant="small-medium"
            tone="muted"
            className="mt-auto self-start rounded-full bg-surface-hover px-2 py-0.5"
          >
            In progress
          </Text>
        </NextLink>
      ) : (
        <button
          type="button"
          {...tourStartingPoint()}
          disabled={isBusy}
          onClick={() => onSend(task)}
          className={cn(
            CARD_CLASSES,
            "cursor-pointer hover:border-brand/40 hover:shadow-md disabled:cursor-default disabled:opacity-60 disabled:hover:border-border-base disabled:hover:shadow-none",
          )}
        >
          <CardTop task={task} isSending={isSending} />
          <Text variant="body-medium" tone="primary" className="text-pretty">
            {task.title}
          </Text>
        </button>
      )}
      <Button
        type="button"
        variant="secondary"
        size="icon-xs"
        withTooltip={false}
        aria-label={`Hide: ${task.title}`}
        disabled={!canHide}
        onClick={() => onHide(task)}
        className="absolute -right-2 -top-2 text-foreground-subtle opacity-0 transition-opacity duration-150 hover:text-foreground-body focus-visible:opacity-100 group-hover/card:opacity-100 [@media(hover:none)]:opacity-100"
      >
        <Icon icon={Cancel01Icon} size={12} />
      </Button>
    </>
  );
}

interface CardTopProps {
  task: SuggestedTask;
  isSending?: boolean;
  muted?: boolean;
}

function CardTop({ task, isSending = false, muted = false }: CardTopProps) {
  const providers = getMentionedProviders(task.title);

  return (
    <span className="flex items-center justify-between">
      <span
        className={cn(
          "flex size-9 items-center justify-center rounded-lg",
          muted
            ? "bg-surface-hover text-foreground-disabled"
            : "bg-accent-surface text-brand",
        )}
      >
        <Icon
          icon={isSending ? Loading03Icon : getPromptIcon(task.icon)}
          size={18}
          className={cn(isSending && "animate-spin")}
          aria-label={isSending ? "Starting" : undefined}
        />
      </span>
      {providers.length > 0 && (
        <span className="flex" aria-hidden>
          {providers.map((provider) => (
            <span
              key={provider}
              className="-ml-1.5 flex size-7 items-center justify-center rounded-full border-2 border-surface bg-surface shadow-sm first:ml-0"
            >
              <IntegrationLogo provider={provider} alt="" size={16} />
            </span>
          ))}
        </span>
      )}
    </span>
  );
}
