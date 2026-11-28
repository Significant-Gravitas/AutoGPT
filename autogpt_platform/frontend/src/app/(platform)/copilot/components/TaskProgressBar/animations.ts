import { duration } from "@/components/styles/motion";

// A newly in-progress loader waits for the completing task's tick + strikethrough
// to play first, so each step reads as a sequence.
export const LOADER_DELAY = 0.3;

// Blur-bridge for the status icon swap (Emil's copy-button pattern).
export const ICON_EXIT = duration.instant;
export const ICON_ENTER = duration.base;
