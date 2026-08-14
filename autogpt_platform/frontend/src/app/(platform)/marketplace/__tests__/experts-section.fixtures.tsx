import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { render } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

export const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "External actions require approval.",
  ],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

export const mariaRichTemplate: Expert = {
  ...mariaTemplate,
  bio: "Maria has run brand launches for a decade and loves a tidy funnel.",
  skills: ["Brand strategy", "SEO"],
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Content Calendar",
      description: "Plans a month of posts",
    },
    {
      id: "wf-2",
      store_listing_version_id: "slv-2",
      library_agent_id: "lib-2",
      graph_id: "graph-2",
      name: "SEO Audit",
      description: null,
    },
  ],
};

export const hiredMaria: Expert = {
  ...mariaTemplate,
  id: "expert-maria",
  is_template: false,
  source_template_id: "template-maria",
};

export function renderMarketplace() {
  return render(
    <>
      <MainMarkeplacePage />
      <Toaster />
    </>,
  );
}
