import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ComponentProps } from "react";
import * as AtomTooltip from "../BaseTooltip";

// What the deleted components/ui/tooltip rendered: the atom's content in a
// portal, with the provider's delay.
const PortalTooltip = {
  ...AtomTooltip,
  TooltipContent: function PortalTooltipContent(
    props: ComponentProps<typeof AtomTooltip.TooltipContent>,
  ) {
    return (
      <AtomTooltip.TooltipPortal>
        <AtomTooltip.TooltipContent {...props} />
      </AtomTooltip.TooltipPortal>
    );
  },
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe.each([
  ["atom (portal)", PortalTooltip],
  ["atom", AtomTooltip],
] as const)("%s tooltip pointer transit", (_, components) => {
  it("shows the next button's tooltip when moving beneath the previous tooltip", async () => {
    const { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } =
      components;
    render(
      <TooltipProvider delayDuration={0}>
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button>First action</button>
          </TooltipTrigger>
          <TooltipContent data-testid="first-tooltip">
            First description
          </TooltipContent>
        </Tooltip>
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button>Second action</button>
          </TooltipTrigger>
          <TooltipContent>Second description</TooltipContent>
        </Tooltip>
      </TooltipProvider>,
    );
    const first = screen.getByRole("button", { name: "First action" });
    const second = screen.getByRole("button", { name: "Second action" });
    vi.spyOn(first, "getBoundingClientRect").mockReturnValue(
      new DOMRect(0, 0, 20, 20),
    );
    fireEvent.pointerMove(first, {
      pointerType: "mouse",
      clientX: 10,
      clientY: 10,
    });
    expect(
      await screen.findByRole("tooltip", { name: "First description" }),
    ).toBeDefined();
    const firstTooltip = screen.getByTestId("first-tooltip");
    vi.spyOn(firstTooltip, "getBoundingClientRect").mockReturnValue(
      new DOMRect(24, 0, 60, 20),
    );
    const position = {
      pointerType: "mouse",
      clientX: 21,
      clientY: 10,
      relatedTarget: second,
    };
    fireEvent.pointerOut(first, position);
    fireEvent.pointerLeave(first, position);
    fireEvent.pointerOver(second, {
      pointerType: "mouse",
      clientX: 35,
      clientY: 10,
      relatedTarget: first,
    });
    fireEvent.pointerMove(second, {
      pointerType: "mouse",
      clientX: 35,
      clientY: 10,
    });
    expect(
      await screen.findByRole("tooltip", { name: "Second description" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("tooltip", { name: "First description" }),
    ).toBeNull();
  });
});

describe("atom tooltip trigger slot", () => {
  it("names the button it renders", () => {
    const { TooltipProvider, Tooltip, TooltipTrigger } = AtomTooltip;
    render(
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger>Trigger</TooltipTrigger>
        </Tooltip>
      </TooltipProvider>,
    );
    expect(
      screen.getByRole("button", { name: "Trigger" }).getAttribute("data-slot"),
    ).toBe("tooltip-trigger");
  });

  it("leaves an asChild child's slot alone", () => {
    const { TooltipProvider, Tooltip, TooltipTrigger } = AtomTooltip;
    render(
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <button data-slot="toggle-chip">Chip</button>
          </TooltipTrigger>
        </Tooltip>
      </TooltipProvider>,
    );
    expect(
      screen.getByRole("button", { name: "Chip" }).getAttribute("data-slot"),
    ).toBe("toggle-chip");
  });
});

// `inheritDelay` is what the deleted components/ui/tooltip did: the delay
// comes from the nearest provider (700ms at the app root, 0 in the sidebar)
// instead of the atom's own 10ms default.
describe("atom tooltip delay", () => {
  function renderTooltip({
    providerDelay,
    inheritDelay,
  }: {
    providerDelay?: number;
    inheritDelay?: boolean;
  }) {
    const { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } =
      AtomTooltip;
    render(
      <TooltipProvider delayDuration={providerDelay}>
        <Tooltip inheritDelay={inheritDelay}>
          <TooltipTrigger asChild>
            <button>Action</button>
          </TooltipTrigger>
          <TooltipContent>Description</TooltipContent>
        </Tooltip>
      </TooltipProvider>,
    );
    fireEvent.pointerMove(screen.getByRole("button", { name: "Action" }), {
      pointerType: "mouse",
    });
  }

  function isOpen() {
    return screen.queryByRole("tooltip") !== null;
  }

  it("takes the provider's default 700ms delay with inheritDelay", () => {
    vi.useFakeTimers();
    try {
      renderTooltip({ inheritDelay: true });
      act(() => vi.advanceTimersByTime(300));
      expect(isOpen()).toBe(false);
      act(() => vi.advanceTimersByTime(450));
      expect(isOpen()).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("takes a provider's 0ms delay with inheritDelay, as in the sidebar", () => {
    vi.useFakeTimers();
    try {
      renderTooltip({ providerDelay: 0, inheritDelay: true });
      act(() => vi.advanceTimersByTime(1));
      expect(isOpen()).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps the atom's own 10ms delay without inheritDelay", () => {
    vi.useFakeTimers();
    try {
      renderTooltip({});
      act(() => vi.advanceTimersByTime(50));
      expect(isOpen()).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});
