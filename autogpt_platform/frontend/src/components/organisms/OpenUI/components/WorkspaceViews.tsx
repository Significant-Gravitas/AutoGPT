import type { ComponentRenderProps } from "@openuidev/react-lang";
import { useIsStreaming, useTriggerAction } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import * as catalog from "@/lib/openui/catalog";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowUpRight01Icon, Idea01Icon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { useOpenUIDisabled } from "../interactionContext";

export function WorkspaceView({
  props,
  renderNode,
}: ComponentRenderProps<z.infer<typeof catalog.Workspace.props>>) {
  return (
    <div className="space-y-5">
      <div className="pb-1">
        <h2 className="text-2xl font-semibold tracking-tight text-zinc-900">
          {props.title}
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-600">
          {props.description}
        </p>
      </div>
      {renderNode(props.children?.slice(0, 8))}
    </div>
  );
}

export function MetricsView({
  props,
  renderNode,
}: ComponentRenderProps<z.infer<typeof catalog.Metrics.props>>) {
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,9rem),1fr))] gap-3">
      {renderNode(props.items?.slice(0, 4))}
    </div>
  );
}

export function MetricView({
  props,
}: ComponentRenderProps<z.infer<typeof catalog.Metric.props>>) {
  return (
    <div className="min-w-0 rounded-xl border border-zinc-200 bg-white p-3.5">
      <p className="text-xs font-medium text-zinc-600">{props.label}</p>
      <p
        className={cn(
          "my-2 break-words font-semibold tracking-tight text-zinc-900",
          (props.value?.length ?? 0) > 8 ? "text-base" : "text-2xl",
        )}
      >
        {props.value}
      </p>
      <p
        className={cn(
          "text-[11px] leading-snug text-zinc-500",
          props.tone === "positive" && "text-green-700",
          props.tone === "warning" && "text-orange-700",
        )}
      >
        {props.detail}
      </p>
    </div>
  );
}

export function InsightView({
  props,
}: ComponentRenderProps<z.infer<typeof catalog.Insight.props>>) {
  return (
    <div
      className={cn(
        "flex gap-3 rounded-xl border border-zinc-200 bg-zinc-50 p-4",
        props.tone === "positive" && "border-green-100 bg-green-50/50",
        props.tone === "warning" && "border-orange-100 bg-orange-50",
      )}
    >
      <Icon
        icon={Idea01Icon}
        size={20}
        className="mt-0.5 shrink-0 text-zinc-600"
      />
      <div>
        <h3 className="text-sm font-medium text-zinc-800">{props.title}</h3>
        <p className="mt-1 text-sm leading-relaxed text-zinc-600">
          {props.body}
        </p>
      </div>
    </div>
  );
}

export function ActionView({
  props,
}: ComponentRenderProps<z.infer<typeof catalog.FollowUp.props>>) {
  const triggerAction = useTriggerAction();
  const isStreaming = useIsStreaming();
  const disabled = useOpenUIDisabled();
  return (
    <Button
      size="small"
      variant="secondary"
      disabled={isStreaming || disabled}
      onClick={() => triggerAction(props.message)}
      rightIcon={<Icon icon={ArrowUpRight01Icon} size={16} />}
      unmask={false}
    >
      {props.label}
    </Button>
  );
}
