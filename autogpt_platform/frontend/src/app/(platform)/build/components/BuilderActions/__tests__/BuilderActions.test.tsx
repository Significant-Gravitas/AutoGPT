import { render, screen } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { describe, expect, test, vi } from "vitest";

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
    return (
      <div
        data-flow-id={props.flowID ?? ""}
        data-flow-version={props.flowVersion ?? ""}
        data-testid="publish-to-marketplace"
      />
    );
  },
}));

import { BuilderActions } from "../BuilderActions";

describe("BuilderActions", () => {
  test("passes flow id and numeric flow version from query state into publish", () => {
    render(
      <NuqsTestingAdapter searchParams="?flowID=graph-1&flowVersion=7">
        <BuilderActions />
      </NuqsTestingAdapter>,
    );

    expect(
      screen.getByTestId("publish-to-marketplace").getAttribute("data-flow-id"),
    ).toBe("graph-1");
    expect(
      screen
        .getByTestId("publish-to-marketplace")
        .getAttribute("data-flow-version"),
    ).toBe("7");
    expect(publishToMarketplaceMock).toHaveBeenCalledWith({
      flowID: "graph-1",
      flowVersion: 7,
    });
  });
});
