"use client";

import type { ReactNode } from "react";
import { TeamRevealEnding } from "./TeamRevealEnding";
import { useTeamReveal } from "./useTeamReveal";

interface Props {
  isLeaving: boolean;
  onMeet: () => void;
  fallback: ReactNode;
}

// The proposed team once the hand-off lands; the plain ending while it is on
// its way, and in its place if it never comes.
export function TeamReveal({ isLeaving, onMeet, fallback }: Props) {
  const { team } = useTeamReveal();
  if (team.length === 0) return <>{fallback}</>;
  return <TeamRevealEnding team={team} isLeaving={isLeaving} onMeet={onMeet} />;
}
