import type { TaskTab } from "../../helpers";

interface EmptyCopy {
  title: string;
  body: string;
}

export const EMPTY_COPY: Record<TaskTab, EmptyCopy> = {
  all: {
    title: "No tasks yet",
    body: "Hand something to your team and it shows up here.",
  },
  needs_you: {
    title: "Nothing needs you",
    body: "When a task waits on your answer or fails, it shows up here.",
  },
  working: {
    title: "Nothing in progress",
    body: "Tasks that are to do or being worked on show up here.",
  },
  done: {
    title: "Nothing finished yet",
    body: "Finished tasks land here.",
  },
  cancelled: {
    title: "Nothing cancelled",
    body: "Tasks you cancel move here.",
  },
};

export const NEW_USER_COPY: EmptyCopy = {
  title: "Hand off your first task",
  body: "A task is work Otto or an expert on your team does for you. Describe what you need and pick who does it.",
};
