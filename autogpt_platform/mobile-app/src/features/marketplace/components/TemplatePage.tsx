import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

import { useTemplatePage } from "../useTemplatePage";
import { HireBar } from "./HireBar";
import { TemplateDetails } from "./TemplateDetails";

interface Props {
  templateID: string;
}

export function TemplatePage({ templateID }: Props) {
  const page = useTemplatePage(templateID);
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();

  function renderBody() {
    if (page.template) {
      return (
        <>
          <ScrollView contentContainerStyle={styles.content}>
            <TemplateDetails template={page.template} />
          </ScrollView>
          <View style={[styles.bar, { paddingBottom: insets.bottom + spacing.md }]}>
            <HireBar
              name={page.template.name}
              isOnTeam={page.isOnTeam}
              isHiring={page.isHiring}
              isOpeningChat={page.isOpeningChat}
              error={page.error}
              onHire={page.hire}
              onChat={page.openChat}
            />
          </View>
        </>
      );
    }
    if (page.isLoading) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    return (
      <View style={styles.missing}>
        <Text style={styles.missingText}>This expert isn&apos;t in the marketplace any more.</Text>
        <Button label="Back to marketplace" variant="secondary" onPress={page.back} />
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        left={<IconButton icon={ArrowLeft01Icon} label="Back" onPress={page.back} />}
      />
      {renderBody()}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.background },
  content: { paddingHorizontal: spacing.xl, paddingBottom: spacing.xl },
  bar: {
    gap: spacing.sm,
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.md,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    backgroundColor: colors.background,
  },
  loading: { marginTop: spacing.xxl },
  missing: { alignItems: "center", gap: spacing.lg, padding: spacing.xl, borderRadius: radius.lg },
  missingText: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
