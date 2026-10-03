import { vi } from "vitest";
import type { ComponentPropsWithoutRef } from "react";

export const mockNextjsModules = () => {
  vi.mock("next/image", () => ({
    __esModule: true,
    default: ({
      fill: _fill,
      priority: _priority,
      quality: _quality,
      placeholder: _placeholder,
      blurDataURL: _blurDataURL,
      loader: _loader,
      ...props
    }: any) => {
      // eslint-disable-next-line jsx-a11y/alt-text, @next/next/no-img-element
      return <img {...props} />;
    },
  }));

  vi.mock("next/headers", () => ({
    cookies: vi.fn(() => ({
      get: vi.fn(() => undefined),
      getAll: vi.fn(() => []),
      set: vi.fn(),
      delete: vi.fn(),
      has: vi.fn(() => false),
    })),
    headers: vi.fn(() => new Headers()),
  }));

  vi.mock("next/dist/server/request/cookies", () => ({
    cookies: vi.fn(() => ({
      get: vi.fn(() => undefined),
      getAll: vi.fn(() => []),
      set: vi.fn(),
      delete: vi.fn(),
      has: vi.fn(() => false),
    })),
  }));

  vi.mock("next/navigation", () => ({
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/marketplace",
    useSearchParams: () => new URLSearchParams(),
    useParams: () => ({}),
  }));

  vi.mock("next/link", async () => {
    const { forwardRef } = await import("react");
    return {
      __esModule: true,
      default: forwardRef<HTMLAnchorElement, ComponentPropsWithoutRef<"a">>(
        function MockLink({ children, href, ...props }, ref) {
          return (
            <a ref={ref} href={href} {...props}>
              {children}
            </a>
          );
        },
      ),
      useLinkStatus: () => ({ pending: false }),
    };
  });
};
