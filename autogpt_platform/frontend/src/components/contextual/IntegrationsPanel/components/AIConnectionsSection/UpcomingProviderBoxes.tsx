import { ProviderBox } from "./ProviderBox";

export function UpcomingProviderBoxes() {
  return (
    <>
      <ProviderBox
        name="Grok"
        logoSrc="/integrations/xai.webp"
        state="coming-soon"
      />
      <ProviderBox
        name="GitHub Copilot"
        logoSrc="/integrations/github.png"
        state="coming-soon"
      />
    </>
  );
}
