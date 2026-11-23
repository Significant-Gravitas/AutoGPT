import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, RefreshControl, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { IconButton } from "@/components/IconButton";
import { ScreenHeader } from "@/components/ScreenHeader";
import { PlanCard } from "@/features/plan/components/PlanCard";
import { usePlanScreen } from "@/features/plan/usePlanScreen";
import { SettingsGroup } from "@/features/settings/components/SettingsGroup";
import { SettingsRow } from "@/features/settings/components/SettingsRow";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

/** The account's plan, read only: nothing here leads to a payment. */
export default function PlanScreen() {
  const plan = usePlanScreen();
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();

  function renderBody() {
    if (plan.isLoading) {
      return (
        <View style={styles.center}>
          <ActivityIndicator color={colors.textTertiary} />
        </View>
      );
    }
    if (plan.isError || !plan.summary) {
      return (
        <View style={styles.center}>
          <Text style={styles.notice}>{"Your plan didn't load. Try again."}</Text>
          <Button label="Try again" variant="secondary" size="small" onPress={plan.refresh} />
        </View>
      );
    }
    return (
      <>
        <PlanCard summary={plan.summary} meters={plan.meters} />
        {plan.balance ? (
          <SettingsGroup title="Credits for workflows">
            <SettingsRow label="Balance" value={plan.balance} />
          </SettingsGroup>
        ) : null}
      </>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        title="Plan"
        left={<IconButton icon={ArrowLeft01Icon} label="Back" onPress={plan.goBack} />}
      />
      <ScrollView
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + spacing.xxl }]}
        refreshControl={
          <RefreshControl refreshing={plan.isRefreshing} onRefresh={plan.refresh} />
        }
      >
        {renderBody()}
      </ScrollView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.menuBackground },
  content: { flexGrow: 1, gap: spacing.xl, padding: spacing.lg },
  center: { flex: 1, alignItems: "center", justifyContent: "center", gap: spacing.lg },
  notice: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
