"use client";

import { Text } from "@/components/atoms/Text/Text";
import { useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** The element above the composer where the open question docks. */
export const QUESTION_DOCK_SLOT_ID = "copilot-question-dock";

interface Props {
  /** The first open question, kept as one line where it was asked. */
  question: string;
  children: ReactNode;
}

/** Shows the question card just above the composer, and leaves one quiet
 *  line in the thread where it was asked. A chat without a composer (a
 *  read-only or side view) keeps the card in place instead. */
export function QuestionDockSlot({ question, children }: Props) {
  // Read again after each commit, so a composer that mounts with the card
  // is found on the next pass.
  const slot = useSyncExternalStore(subscribeToNothing, findSlot, noSlot);

  if (!slot) return <>{children}</>;
  return (
    <>
      <Text variant="inbox" tone="quiet" className="text-pretty">
        Asked: {question}
        <Text as="span" variant="inbox" tone="faint">
          {" · waiting"}
        </Text>
      </Text>
      {createPortal(<div className="mb-2">{children}</div>, slot)}
    </>
  );
}

function subscribeToNothing() {
  return () => {};
}

function findSlot() {
  return document.getElementById(QUESTION_DOCK_SLOT_ID);
}

function noSlot() {
  return null;
}
