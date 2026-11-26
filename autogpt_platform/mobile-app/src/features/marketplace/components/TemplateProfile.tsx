import { View } from "react-native";

import { avatarTint } from "@/features/team/helpers";
import type { ExpertTemplate } from "@/lib/api/experts";
import { createStyles } from "@/lib/theme";

import { firstName } from "../helpers";
import { profileStats } from "../profileHelpers";
import { AboutText } from "./profile/AboutText";
import { DayOneSteps } from "./profile/DayOneSteps";
import { HireTermsCard } from "./profile/HireTermsCard";
import { ProtectedRules } from "./profile/ProtectedRules";
import { ServiceGrid } from "./profile/ServiceGrid";
import { SkillList } from "./profile/SkillList";
import { TemplateHero } from "./profile/TemplateHero";
import { WorkflowList } from "./profile/WorkflowList";

interface Props {
  template: ExpertTemplate;
  /** The services its workflows sign in to that you would connect. */
  services: string[];
}

/** The web's expert page, in its order. A section with nothing to show is left out. */
export function TemplateProfile({ template, services }: Props) {
  const styles = useStyles();
  const name = firstName(template.name);
  const accent = avatarTint(template);
  const workflows = template.workflows ?? [];

  return (
    <View>
      <TemplateHero template={template} stats={profileStats(template, services.length)} />
      <View style={styles.body}>
        <HireTermsCard
          name={name}
          workflowCount={workflows.length}
          weeklyBudget={template.weekly_budget}
        />
        <DayOneSteps name={name} items={template.day_one ?? []} accent={accent} />
        {template.bio ? <AboutText key={template.id} text={template.bio} /> : null}
        <SkillList skills={template.bundled_skills} accent={accent} />
        <ServiceGrid name={name} providers={services} />
        <ProtectedRules rules={template.protected_soul_rules ?? []} />
        <WorkflowList name={name} workflows={workflows} />
      </View>
    </View>
  );
}

const useStyles = createStyles(() => ({
  body: { paddingHorizontal: 14, paddingTop: 26, gap: 30 },
}));
