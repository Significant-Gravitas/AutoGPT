import { useLocalSearchParams } from "expo-router";

import { TemplatePage } from "@/features/marketplace/components/TemplatePage";

export default function TemplateScreen() {
  const { templateId } = useLocalSearchParams<{ templateId: string }>();
  // Keyed: the drawer keeps this screen mounted, and another expert starts fresh.
  return <TemplatePage key={templateId} templateID={templateId} />;
}
