"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import type { CSSProperties } from "react";
import { HandoffDocument } from "./HandoffDocument";
import { useTaskHandoff } from "./useTaskHandoff";

const POP = { type: "spring", duration: 0.55, bounce: 0.3 } as const;
const BUBBLE_DELAY_S = 0.1;
const DOTS = [
  { x: "-10px", delay: "180ms", size: "size-1.5" },
  { x: "6px", delay: "260ms", size: "size-2" },
  { x: "16px", delay: "340ms", size: "size-1" },
];

interface Props {
  taskID: string;
}

/** A finished task, handed over in the chat by whoever did it: their
 *  portrait pops in and a speech bubble, in the fill of your own messages,
 *  grows out of it with the result in their words and the document that
 *  holds the rest. Otto's own line follows it as plain text. A result met
 *  on reopening the chat is drawn still. */
export function TaskHandoff({ taskID }: Props) {
  const { isLoading, handoff } = useTaskHandoff(taskID);
  const prefersReducedMotion = useReducedMotion();

  if (isLoading) {
    return (
      <div aria-hidden className="flex items-end gap-2.5 py-1.5">
        <span className="size-9 shrink-0 rounded-lg bg-surface-tray" />
        <span className="h-36 flex-1 rounded-2xl rounded-bl-md bg-surface-tray" />
      </div>
    );
  }
  if (!handoff) return null;

  const isPlayed = handoff.isFresh && !prefersReducedMotion;
  const from = (scale: number) =>
    isPlayed ? { opacity: 0, scale, y: 6 } : false;
  const settled = { opacity: 1, scale: 1, y: 0 };

  return (
    <section
      aria-label={`${handoff.owner.name} finished ${handoff.title}`}
      className="flex items-end gap-2.5 py-1.5"
    >
      <motion.span
        initial={from(0.6)}
        animate={settled}
        transition={POP}
        className="relative shrink-0"
      >
        <ExpertAvatar
          name={handoff.owner.name}
          avatarUrl={handoff.owner.avatarUrl}
          size={36}
          shape="soft"
        />
        {isPlayed
          ? DOTS.map((dot) => (
              <span
                key={dot.x}
                aria-hidden
                className={`pointer-events-none absolute -top-1 left-1/2 animate-handoff-dot rounded-full bg-foreground-ink/15 ${dot.size}`}
                style={
                  {
                    "--handoff-x": dot.x,
                    "--handoff-delay": dot.delay,
                  } as CSSProperties
                }
              />
            ))
          : null}
      </motion.span>
      <motion.div
        initial={from(0.9)}
        animate={settled}
        transition={{ ...POP, delay: isPlayed ? BUBBLE_DELAY_S : 0 }}
        className="flex min-w-0 flex-1 origin-bottom-left flex-col gap-1.5 rounded-2xl rounded-bl-md bg-surface-tray p-1.5"
      >
        <div className="flex flex-col gap-1 px-2.5 pt-1.5 pb-1">
          <span className="flex flex-wrap items-baseline gap-x-2">
            <Text as="span" variant="inbox-medium" tone="ink">
              {handoff.owner.name}
            </Text>
            <Text as="span" variant="small" tone="quiet">
              {[handoff.role, handoff.took && `finished in ${handoff.took}`]
                .filter(Boolean)
                .join(" · ")}
            </Text>
            <button
              type="button"
              onClick={handoff.openTask}
              className="ml-auto flex items-center gap-0.5 rounded-sm text-caption text-foreground-quiet transition-colors duration-100 ease-out outline-none hover:text-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox"
            >
              See task
              <Icon icon={ArrowUpRight01Icon} size={12} tone="inherit" />
            </button>
          </span>
          {handoff.headline ? (
            <Text variant="body-medium" tone="ink" className="text-pretty">
              {handoff.headline}
            </Text>
          ) : null}
          <Text
            variant="body"
            tone={handoff.headline ? "text" : "ink"}
            className="text-pretty"
          >
            {handoff.body}
          </Text>
        </div>
        {handoff.document ? (
          <HandoffDocument
            title={handoff.document.title}
            kind={handoff.document.kind}
            onOpen={handoff.openDocument}
          />
        ) : null}
      </motion.div>
    </section>
  );
}
