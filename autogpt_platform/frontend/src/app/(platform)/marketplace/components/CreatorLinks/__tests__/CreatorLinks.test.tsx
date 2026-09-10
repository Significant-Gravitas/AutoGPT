import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { CreatorLinks } from "../CreatorLinks";

afterEach(cleanup);

// Each social host maps to its own brand glyph. A wrong or undefined icon
// import renders an empty <svg> rather than throwing, so compare the drawn
// geometry instead of asserting the element exists.
function glyph(root: HTMLElement) {
  const svg = root.querySelector("svg");
  if (!svg) return "";
  return Array.from(
    svg.querySelectorAll("path,circle,rect,line,polyline,ellipse"),
  )
    .map((n) => n.getAttribute("d") ?? n.outerHTML)
    .join("|");
}

function renderLink(url: string) {
  const { container, unmount } = render(<CreatorLinks links={[url]} />);
  const g = glyph(container);
  unmount();
  return g;
}

describe("CreatorLinks", () => {
  it("renders nothing when there are no links", () => {
    const { container } = render(<CreatorLinks links={[]} />);
    expect(container.innerHTML).toBe("");
  });

  it("labels each link with its hostname, stripped of scheme and www", () => {
    render(
      <CreatorLinks
        links={["https://www.github.com/acme", "example.com/page"]}
      />,
    );

    expect(screen.getByText("github.com")).toBeDefined();
    expect(screen.getByText("example.com")).toBeDefined();
  });

  it("gives each recognised social host its own brand glyph", () => {
    const hosts = [
      "https://facebook.com/acme",
      "https://x.com/acme",
      "https://instagram.com/acme",
      "https://linkedin.com/in/acme",
      "https://github.com/acme",
      "https://youtube.com/@acme",
      "https://tiktok.com/@acme",
    ];

    const drawn = hosts.map(renderLink);
    drawn.forEach((g) => expect(g).toBeTruthy());
    expect(new Set(drawn).size).toBe(drawn.length);
  });

  it("treats twitter.com and x.com as the same brand", () => {
    expect(renderLink("https://twitter.com/acme")).toBe(
      renderLink("https://x.com/acme"),
    );
  });

  it("matches subdomains of a known host", () => {
    expect(renderLink("https://www.youtube.com/@acme")).toBe(
      renderLink("https://youtube.com/@acme"),
    );
  });

  it("falls back to the globe glyph for unknown hosts", () => {
    const globe = renderLink("https://example.com");
    expect(globe).toBeTruthy();
    expect(renderLink("https://some-blog.dev")).toBe(globe);
    // A host that merely contains a brand name must not borrow its glyph.
    expect(renderLink("https://notgithub.com")).toBe(globe);
  });

  it("falls back to the globe glyph when the URL cannot be parsed", () => {
    const unparseable = renderLink("http://");
    expect(unparseable).toBeTruthy();
    expect(unparseable).toBe(renderLink("https://example.com"));
  });
});
