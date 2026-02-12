import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainCreatorPage } from "../MainCreatorPage";

const defaultParams = {
  creator: "test-creator",
};

describe("MainCreatorPage - Rendering", () => {
  test("renders creator description", async () => {
    render(<MainCreatorPage params={defaultParams} />);
    expect(
      await screen.findByTestId("creator-description"),
    ).toBeInTheDocument();
  });

  test("renders breadcrumbs with marketplace link", async () => {
    render(<MainCreatorPage params={defaultParams} />);

    expect(
      await screen.findByRole("link", { name: /marketplace/i }),
    ).toBeInTheDocument();
  });

  test("renders about section", async () => {
    render(<MainCreatorPage params={defaultParams} />);

    expect(await screen.findByText("About")).toBeInTheDocument();
  });

  test("renders agents by creator section", async () => {
    render(<MainCreatorPage params={defaultParams} />);

    expect(
      await screen.findByText(/Agents by/i, { exact: false }),
    ).toBeInTheDocument();
  });
});
