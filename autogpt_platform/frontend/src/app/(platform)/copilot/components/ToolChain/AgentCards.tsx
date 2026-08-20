"use client";

import {
  LinkSquare01Icon,
  PlayIcon,
  Robot01Icon,
  StarIcon,
  WrenchIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CARD, HALF, StatusPill } from "./ResultCards";
import { inline, resultItemKey, str } from "./resultHelpers";

function agentHref(agent: Record<string, unknown>): string | null {
  const id = str(agent, "id");
  if (!id) return null;
  if (str(agent, "source") === "library")
    return `/library/agents/${encodeURIComponent(id)}`;
  const [creator, slug, ...rest] = id.split("/");
  if (!creator || !slug || rest.length > 0) return null;
  return `/marketplace/agent/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`;
}

interface CardLinkProps {
  href: string;
  label: string;
}

interface AgentListCardProps {
  agents: Record<string, unknown>[];
}

interface OutputCardProps {
  output: Record<string, unknown>;
}

function CardLink({ href, label }: CardLinkProps) {
  return (
    <Link
      href={href}
      aria-label={label}
      className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
    >
      <Icon icon={LinkSquare01Icon} size={14} />
    </Link>
  );
}

export function AgentListCard({ agents }: AgentListCardProps) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-2">
      {agents.map((agent, i) => {
        const href = agentHref(agent);
        const runs = typeof agent.runs === "number" ? agent.runs : null;
        const rating = typeof agent.rating === "number" ? agent.rating : null;
        const subtitle = str(agent, "description");
        return (
          <div
            key={resultItemKey(agent, i)}
            className={CARD + " flex items-start gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <Icon icon={Robot01Icon} size={15} className="text-zinc-600" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <p className="min-w-0 truncate text-[13px] font-medium text-zinc-800">
                  {str(agent, "name", "agent_name") ?? inline(agent)}
                </p>
                {str(agent, "creator") && (
                  <span className="shrink-0 truncate text-[11px] text-zinc-400">
                    by {str(agent, "creator")}
                  </span>
                )}
              </div>
              {subtitle && (
                <p className="truncate text-xs text-zinc-500">{subtitle}</p>
              )}
              {(runs !== null || rating !== null) && (
                <div className="mt-1 flex items-center gap-2.5 text-[11px] text-zinc-400">
                  {runs !== null && (
                    <span className="flex items-center gap-1">
                      <Icon icon={PlayIcon} size={10} />
                      {runs.toLocaleString()} runs
                    </span>
                  )}
                  {rating !== null && (
                    <span className="flex items-center gap-1">
                      <Icon
                        icon={StarIcon}
                        size={10}
                        className="text-amber-400"
                      />
                      {rating.toFixed(1)}
                    </span>
                  )}
                </div>
              )}
            </div>
            {href && <CardLink href={href} label="Open agent" />}
          </div>
        );
      })}
    </div>
  );
}

export function AgentSavedCard({ output }: OutputCardProps) {
  const name = str(output, "agent_name", "name", "graph_name") ?? "Agent";
  const version = output.graph_version;
  const libraryLink =
    str(output, "library_agent_link") ??
    (str(output, "library_agent_id")
      ? `/library/agents/${str(output, "library_agent_id")}`
      : null);
  const builderLink = str(output, "agent_page_link");
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon icon={Robot01Icon} size={15} className="text-zinc-600" />
      </div>
      <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-800">
        {name}
      </p>
      {typeof version === "number" && (
        <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
          v{version}
        </span>
      )}
      {builderLink && (
        <Link
          href={builderLink}
          aria-label="Open in builder"
          className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <Icon icon={WrenchIcon} size={14} />
        </Link>
      )}
      {libraryLink && <CardLink href={libraryLink} label="Open in library" />}
    </div>
  );
}

export function AgentPreviewCard({ output }: OutputCardProps) {
  const name = str(output, "agent_name", "name") ?? "Agent preview";
  const count =
    typeof output.node_count === "number" ? output.node_count : null;
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon icon={Robot01Icon} size={15} className="text-zinc-600" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-800">{name}</p>
        {str(output, "description", "message") && (
          <p className="truncate text-xs text-zinc-500">
            {str(output, "description", "message")}
          </p>
        )}
      </div>
      {count !== null && (
        <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
          {count} block{count === 1 ? "" : "s"}
        </span>
      )}
    </div>
  );
}

export function SubSessionCard({ output }: OutputCardProps) {
  const status = str(output, "status");
  const response = str(output, "response");
  const link = str(output, "sub_autopilot_session_link");
  const elapsed =
    typeof output.elapsed_seconds === "number" ? output.elapsed_seconds : null;
  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      <div className="flex items-center gap-2.5">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
          <Icon icon={Robot01Icon} size={15} className="text-zinc-600" />
        </div>
        <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-800">
          Sub-AutoPilot
        </p>
        {elapsed !== null && (
          <span className="shrink-0 text-[11px] text-zinc-400">
            {elapsed >= 60
              ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`
              : `${Math.round(elapsed)}s`}
          </span>
        )}
        {status && <StatusPill status={status} />}
        {link && <CardLink href={link} label="Open sub-session" />}
      </div>
      {response && (
        <p className="mt-1.5 line-clamp-2 pl-9 text-xs text-zinc-500">
          {response}
        </p>
      )}
    </div>
  );
}
