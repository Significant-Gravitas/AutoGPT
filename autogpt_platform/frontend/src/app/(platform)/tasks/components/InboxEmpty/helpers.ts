import type { InboxTab } from "../../helpers";

interface EmptyCopy {
  title: string;
  body: string;
}

export const EMPTY_COPY: Record<InboxTab, EmptyCopy> = {
  all: {
    title: "No tasks yet",
    body: "Hand something to your team and it shows up here while they work.",
  },
  needs_you: {
    title: "Nothing needs you",
    body: "When your team has a question or gets stuck, it waits here.",
  },
  working: {
    title: "Nothing in progress",
    body: "Tasks your team is working on show up here, live.",
  },
  done: {
    title: "Nothing finished yet",
    body: "Finished tasks land here with what came out of them.",
  },
  cancelled: {
    title: "Nothing cancelled",
    body: "Tasks you cancel move here, with what was done before they stopped.",
  },
};

export const FILTERED_COPY: EmptyCopy = {
  title: "No tasks match",
  body: "Try removing a filter to see more.",
};
