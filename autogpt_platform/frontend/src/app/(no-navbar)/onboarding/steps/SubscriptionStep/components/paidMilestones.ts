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
  const renews = getRenewalDate(now, cadence);
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

// Stripe renews on the same day of the next period, or on its last day when
// that month is shorter (31 Jan renews on 28 Feb, 29 Feb on 28 Feb).
export function getRenewalDate(now: Date, cadence: "month" | "year") {
  const months = cadence === "year" ? 12 : 1;
  const target = new Date(now.getFullYear(), now.getMonth() + months, 1);
  const lastDay = new Date(
    target.getFullYear(),
    target.getMonth() + 1,
    0,
  ).getDate();
  target.setDate(Math.min(now.getDate(), lastDay));
  target.setHours(now.getHours(), now.getMinutes(), now.getSeconds());
  return target;
}
