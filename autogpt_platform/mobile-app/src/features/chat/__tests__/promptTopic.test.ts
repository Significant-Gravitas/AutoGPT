import {
  Calendar03Icon,
  ChartLineData01Icon,
  File01Icon,
  Mail01Icon,
  PencilEdit02Icon,
  Search01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";

import { promptTopic } from "@/features/chat/promptTopic";
import { palette } from "@/lib/expertPalette";

describe("promptTopic", () => {
  it.each([
    ["Draft a follow-up email", Mail01Icon],
    ["What reporting features can AutoGPT enhance?", ChartLineData01Icon],
    ["How can AutoGPT assist with data analysis?", ChartLineData01Icon],
    ["Plan my week", Calendar03Icon],
    ["Research a topic for me", Search01Icon],
    ["Sum up a long document", File01Icon],
    ["Write a launch post", PencilEdit02Icon],
  ])("gives %p its own icon", (prompt, icon) => {
    expect(promptTopic(prompt).icon).toBe(icon);
  });

  it("falls back to Otto's sparkle and colour", () => {
    expect(promptTopic("Surprise me")).toEqual({
      icon: SparklesIcon,
      tint: palette.otto.hex,
    });
  });
});
