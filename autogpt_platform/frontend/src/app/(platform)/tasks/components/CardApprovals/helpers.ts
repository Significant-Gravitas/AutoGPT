import type { CardApproval } from "@/app/api/__generated__/models/cardApproval";

export function cardLine(card: CardApproval) {
  const verb = card.kind === "hire" ? "Hire" : "Create";
  return card.role
    ? `${verb} ${card.name} · ${card.role}`
    : `${verb} ${card.name}`;
}

export function approveLabel(count: number) {
  if (count <= 1) return "Approve";
  if (count === 2) return "Approve both";
  return `Approve all ${count}`;
}
