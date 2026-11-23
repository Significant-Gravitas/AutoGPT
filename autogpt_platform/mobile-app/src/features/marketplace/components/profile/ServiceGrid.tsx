import { Text, View } from "react-native";

import { ProviderTile } from "@/features/connections/components/ProviderTile";
import { logoID } from "@/features/connections/helpers";
import { providerDisplayName } from "@/features/connections/listHelpers";
import { createStyles, radius, spacing } from "@/lib/theme";

import { ProfileSection } from "./ProfileSection";

interface Props {
  name: string;
  providers: string[];
}

/** The services the expert's workflows use, two to a row. */
export function ServiceGrid({ name, providers }: Props) {
  const styles = useStyles();
  if (providers.length === 0) return null;

  return (
    <ProfileSection
      title={`Services ${name} can work with`}
      description={`${name} will ask you to connect these the first time they are needed — nothing is shared until you do.`}
    >
      <View style={styles.grid}>
        {providers.map((provider) => {
          const label = providerDisplayName(provider);
          return (
            <View key={provider} style={styles.tile}>
              <ProviderTile name={label} logoID={logoID(provider)} size={36} />
              <Text numberOfLines={2} style={styles.label}>
                {label}
              </Text>
            </View>
          );
        })}
      </View>
    </ProfileSection>
  );
}

const useStyles = createStyles((colors) => ({
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  tile: {
    // Two to a row, less half the gap between them.
    width: "48.5%",
    flexDirection: "row",
    alignItems: "center",
    gap: 11,
    padding: spacing.md,
    borderRadius: radius.lg,
    backgroundColor: colors.panelCard,
  },
  label: { flex: 1, fontSize: 15, lineHeight: 19, fontWeight: "600", color: colors.text },
}));
