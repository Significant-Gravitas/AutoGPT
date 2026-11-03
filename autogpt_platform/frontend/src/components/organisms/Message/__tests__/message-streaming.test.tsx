import { cleanup, render, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import type { StreamdownProps } from "streamdown";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageResponse } from "../Message";

const streamdownProps = vi.hoisted(() => [] as StreamdownProps[]);

vi.mock("streamdown", async (importOriginal) => {
  const actual = await importOriginal<typeof import("streamdown")>();
  function RecordingStreamdown(
    props: ComponentProps<typeof actual.Streamdown>,
  ) {
    streamdownProps.push(props);
    return <actual.Streamdown {...props} />;
  }
  return { ...actual, Streamdown: RecordingStreamdown };
});

const TABLE = "| Account | Status |\n| --- | --- |\n| Work | ok |";

function Strong({ children }: { children?: React.ReactNode }) {
  return <strong data-testid="custom-strong">{children}</strong>;
}

const CUSTOM_COMPONENTS = { strong: Strong };

describe("MessageResponse while a reply streams in", () => {
  afterEach(() => {
    cleanup();
    streamdownProps.length = 0;
  });

  it("hands Streamdown the same components, plugins and link safety on every flush", async () => {
    const { rerender, container } = render(
      <MessageResponse components={CUSTOM_COMPONENTS}>Hello</MessageResponse>,
    );
    rerender(
      <MessageResponse components={CUSTOM_COMPONENTS}>
        Hello **world**
      </MessageResponse>,
    );
    await waitFor(() =>
      expect(
        container.querySelector('[data-testid="custom-strong"]'),
      ).toBeTruthy(),
    );

    expect(streamdownProps.length).toBeGreaterThanOrEqual(2);
    const [first, ...rest] = streamdownProps;
    for (const props of rest) {
      expect(props.components).toBe(first.components);
      expect(props.plugins).toBe(first.plugins);
      expect(props.linkSafety).toBe(first.linkSafety);
    }
  });

  it("keeps the same props across separate messages without custom components", () => {
    render(<MessageResponse>First reply</MessageResponse>);
    render(<MessageResponse>Second reply</MessageResponse>);

    const [first, second] = streamdownProps;
    expect(second.components).toBe(first.components);
    expect(second.plugins).toBe(first.plugins);
    expect(second.linkSafety).toBe(first.linkSafety);
  });

  it("keeps the table and its scroll position while text streams in after it", async () => {
    const { rerender, container } = render(
      <MessageResponse>{TABLE}</MessageResponse>,
    );
    await waitFor(() => expect(container.querySelector("table")).toBeTruthy());
    const wrapper = container.querySelector("table")?.parentElement;
    expect(wrapper?.classList.contains("overflow-x-auto")).toBe(true);
    wrapper!.scrollLeft = 40;

    rerender(
      <MessageResponse>{`${TABLE}\n\nBoth accounts are connected.`}</MessageResponse>,
    );
    await waitFor(() =>
      expect(container.textContent).toContain("Both accounts are connected."),
    );

    expect(container.querySelector("table")?.parentElement).toBe(wrapper);
    expect(wrapper!.scrollLeft).toBe(40);
  });

  it("lets the caller's components override the default table wrapper", async () => {
    function PlainTable({ children }: { children?: React.ReactNode }) {
      return <table data-testid="plain-table">{children}</table>;
    }
    const { container } = render(
      <MessageResponse components={{ table: PlainTable }}>
        {TABLE}
      </MessageResponse>,
    );
    await waitFor(() =>
      expect(
        container.querySelector('[data-testid="plain-table"]'),
      ).toBeTruthy(),
    );
    expect(container.querySelector(".overflow-x-auto")).toBeNull();
  });

  it("picks up a component the caller replaced in place on the same map", async () => {
    function FirstTable({ children }: { children?: React.ReactNode }) {
      return <table data-testid="first-table">{children}</table>;
    }
    function SecondTable({ children }: { children?: React.ReactNode }) {
      return <table data-testid="second-table">{children}</table>;
    }
    const components: Record<string, typeof FirstTable> = {
      table: FirstTable,
    };
    const { container, rerender } = render(
      <MessageResponse components={components}>{TABLE}</MessageResponse>,
    );
    await waitFor(() =>
      expect(
        container.querySelector('[data-testid="first-table"]'),
      ).toBeTruthy(),
    );

    components.table = SecondTable;
    rerender(
      <MessageResponse components={components}>
        {`${TABLE}\n\nDone.`}
      </MessageResponse>,
    );

    await waitFor(() =>
      expect(
        container.querySelector('[data-testid="second-table"]'),
      ).toBeTruthy(),
    );
  });
});
