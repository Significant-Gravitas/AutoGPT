import {
  Calendar03Icon,
  Settings02Icon,
  SquareUnlock02Icon,
} from "@hugeicons/core-free-icons";
import type { Milestone } from "./Timeline";
import { formatDay } from "./TrialWelcome/helpers";

interface Args {
  price: string;
  cadence: "month" | "year";
  now?: Date;
}

export function getPaidMilestones({
  price,
  cadence,
  now = new Date(),
}: Args): Milestone[] {
  const renews = new Date(now);
  if (cadence === "year") renews.setFullYear(now.getFullYear() + 1);
  else renews.setMonth(now.getMonth() + 1);
  return [
    {
      key: "today",
      icon: SquareUnlock02Icon,
      when: "Today",
      title: `You pay ${price}`,
      detail: "Plus any applicable tax. Full access starts straight away.",
    },
    {
      key: "renew",
      icon: Calendar03Icon,
      when: formatDay(renews),
      date: renews,
      title: "Your plan renews",
      detail: `${price} a ${cadence} until you cancel.`,
    },
    {
      key: "cancel",
      icon: Settings02Icon,
      when: "Any time",
      title: "Cancel in Settings",
      detail: "You keep access until the end of the period you've paid for.",
    },
  ];
}
