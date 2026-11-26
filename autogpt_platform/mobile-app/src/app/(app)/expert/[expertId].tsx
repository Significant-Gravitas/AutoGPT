import {
  ArrowLeft01Icon,
  RefreshIcon,
  UserEdit01Icon,
  UserGroupIcon,
  UserRemove01Icon,
  Wallet01Icon,
} from "@hugeicons/core-free-icons";
import { router, useLocalSearchParams } from "expo-router";
import { ActivityIndicator, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { Icon } from "@/components/Icon";
import { IconButton } from "@/components/IconButton";
import { PanelEmptyState } from "@/components/PanelEmptyState";
import { ScreenHeader } from "@/components/ScreenHeader";
import { Toast } from "@/components/Toast";
import { BudgetSheet } from "@/features/expertManage/components/BudgetSheet";
import { EditProfileSheet } from "@/features/expertManage/components/EditProfileSheet";
import { FireExpertSheet } from "@/features/expertManage/components/FireExpertSheet";
import { ModePicker } from "@/features/expertManage/components/ModePicker";
import { ResetExpertSheet } from "@/features/expertManage/components/ResetExpertSheet";
import { formatBudget, formatDollars } from "@/features/expertManage/helpers";
import { useManageExpert } from "@/features/expertManage/useManageExpert";
import { SettingsGroup } from "@/features/settings/components/SettingsGroup";
import { SettingsRow } from "@/features/settings/components/SettingsRow";
import { createStyles, spacing, type, useColors } from "@/lib/theme";

export default function ManageExpertScreen() {
  const { expertId, from } = useLocalSearchParams<{ expertId: string; from?: string }>();
  // Keyed: the drawer keeps this screen mounted; another expert or origin starts fresh.
  return (
    <ManageExpert key={`${expertId}:${from ?? ""}`} routeExpertID={expertId} routeFrom={from} />
  );
}

interface Props {
  routeExpertID: unknown;
  routeFrom: unknown;
}

function ManageExpert({ routeExpertID, routeFrom }: Props) {
  const manage = useManageExpert({ routeExpertID, routeFrom });
  const { expert } = manage;
  const styles = useStyles();
  const colors = useColors();
  const insets = useSafeAreaInsets();

  function openTeam() {
    router.navigate("/team");
  }

  function renderBody() {
    if (expert) {
      return (
        <ScrollView
          contentContainerStyle={[
            styles.content,
            { paddingBottom: insets.bottom + spacing.xxl },
          ]}
        >
          <SettingsGroup title="Profile">
            <SettingsRow
              icon={UserEdit01Icon}
              label="Name and instructions"
              trailing="page"
              onPress={manage.openProfile}
            />
          </SettingsGroup>
          {manage.showApprovals ? (
            <SettingsGroup title="Approvals">
              <ModePicker value={expert.autopilot_mode} onChoose={manage.chooseMode} />
            </SettingsGroup>
          ) : null}
          <SettingsGroup title="Spending">
            <SettingsRow
              icon={Wallet01Icon}
              label="Weekly budget"
              value={formatBudget(expert.weekly_budget)}
              detail={`${formatDollars(expert.weekly_spend)} spent this week`}
              trailing="page"
              onPress={manage.openBudget}
            />
          </SettingsGroup>
          <SettingsGroup>
            <SettingsRow
              icon={RefreshIcon}
              label="Reset"
              isDestructive
              onPress={manage.openReset}
            />
            <SettingsRow
              icon={UserRemove01Icon}
              label="Fire"
              isDestructive
              onPress={manage.openFire}
            />
          </SettingsGroup>
        </ScrollView>
      );
    }
    if (manage.isLoading) {
      return <ActivityIndicator style={styles.loading} color={colors.textTertiary} />;
    }
    if (manage.isLoadError) {
      return (
        <View style={styles.missing}>
          <Text style={styles.missingText}>Couldn&apos;t load this expert.</Text>
          <Button label="Try again" variant="secondary" onPress={manage.retry} />
        </View>
      );
    }
    return (
      <PanelEmptyState
        tiles={[
          <Icon key="profile" icon={UserEdit01Icon} size={22} color={colors.sky} />,
          <Icon key="budget" icon={Wallet01Icon} size={22} color={colors.success} />,
          <Icon key="team" icon={UserGroupIcon} size={22} color={colors.brand} />,
        ]}
        title="This expert is no longer on your team."
        description="Open your team to see who's on it now."
      >
        <Button label="Your team" variant="secondary" size="small" onPress={openTeam} />
      </PanelEmptyState>
    );
  }

  return (
    <View style={styles.screen}>
      <ScreenHeader
        title={expert?.name ?? ""}
        left={<IconButton icon={ArrowLeft01Icon} label="Back" onPress={manage.goBack} />}
      />
      {renderBody()}
      {expert ? (
        <>
          <EditProfileSheet
            isOpen={manage.sheet === "profile"}
            saved={{
              name: expert.name,
              identity: expert.identity,
              voice_preferences: expert.voice_preferences,
              boundaries: expert.boundaries,
            }}
            onClose={manage.closeSheet}
            onSave={manage.saveProfile}
          />
          <BudgetSheet
            isOpen={manage.sheet === "budget"}
            saved={expert.weekly_budget}
            onClose={manage.closeSheet}
            onSave={manage.saveBudget}
          />
          <ResetExpertSheet
            expertID={expert.id}
            name={expert.name}
            hasTemplate={expert.source_template_id !== null}
            isOpen={manage.sheet === "reset"}
            onClose={manage.closeSheet}
            onReset={manage.forgetOrigin}
          />
          <FireExpertSheet
            expertID={expert.id}
            name={expert.name}
            isOpen={manage.sheet === "fire"}
            onClose={manage.closeSheet}
          />
        </>
      ) : null}
      {manage.toast ? <Toast toast={manage.toast} /> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.menuBackground },
  content: { gap: spacing.xl, padding: spacing.lg },
  loading: { marginTop: spacing.xxl },
  missing: { alignItems: "center", gap: spacing.lg, padding: spacing.xl },
  missingText: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
