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
    body: "When an expert has a question or something to review, it waits here.",
  },
  running: {
    title: "Nothing running right now",
    body: "Tasks your team is working on show up here, live.",
  },
  waiting: {
    title: "Nothing waiting",
    body: "Tasks that are queued or paused show up here.",
  },
  done: {
    title: "Nothing finished yet",
    body: "Finished tasks land here with what came out of them.",
  },
  failed: {
    title: "No failures",
    body: "If a task runs into a problem, you'll find it here.",
  },
};

export const FILTERED_COPY: EmptyCopy = {
  title: "No tasks match",
  body: "Try removing a filter to see more.",
};
