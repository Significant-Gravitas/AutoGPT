import { render } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, test, vi } from "vitest";

const publishToMarketplaceMock = vi.hoisted(() => vi.fn());

vi.mock("../components/AgentOutputs/AgentOutputs", () => ({
  AgentOutputs: () => <div />,
}));

vi.mock("../components/RunGraph/RunGraph", () => ({
  RunGraph: () => <div />,
}));

vi.mock("../components/ScheduleGraph/ScheduleGraph", () => ({
  ScheduleGraph: () => <div />,
}));

vi.mock("../components/PublishToMarketplace/PublishToMarketplace", () => ({
  PublishToMarketplace: (props: {
    flowID: string | null;
    flowVersion: number | null;
  }) => {
    publishToMarketplaceMock(props);
    return <div data-testid="publish-to-marketplace" />;
  },
}));

import { BuilderActions } from "../BuilderActions";

function renderBuilderActions(searchParams: string) {
  render(
    <NuqsTestingAdapter searchParams={searchParams}>
      <BuilderActions />
    </NuqsTestingAdapter>,
  );
}

describe("BuilderActions", () => {
  beforeEach(() => {
    publishToMarketplaceMock.mockClear();
  });

  test("passes flow id and numeric flow version from query state into publish", () => {
    renderBuilderActions("?flowID=graph-1&flowVersion=7");

    expect(publishToMarketplaceMock).toHaveBeenCalledWith({
      flowID: "graph-1",
      flowVersion: 7,
    });
  });

  test("passes null flow version when the query param is missing", () => {
    renderBuilderActions("?flowID=graph-1");

    expect(publishToMarketplaceMock).toHaveBeenCalledWith({
      flowID: "graph-1",
      flowVersion: null,
    });
  });

  test("passes null flow id when the query param is missing", () => {
    renderBuilderActions("?flowVersion=7");

    expect(publishToMarketplaceMock).toHaveBeenCalledWith({
      flowID: null,
      flowVersion: 7,
    });
  });

  test("passes null values when query params are absent", () => {
    renderBuilderActions("");

    expect(publishToMarketplaceMock).toHaveBeenCalledWith({
      flowID: null,
      flowVersion: null,
    });
  });

  test("passes null flow version when the query param is invalid", () => {
    renderBuilderActions("?flowID=graph-1&flowVersion=abc");

    expect(publishToMarketplaceMock).toHaveBeenCalledWith({
      flowID: "graph-1",
      flowVersion: null,
    });
  });
});
