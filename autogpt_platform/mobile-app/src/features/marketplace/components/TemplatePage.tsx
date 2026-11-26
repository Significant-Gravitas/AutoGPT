import { ArrowLeft01Icon, Share08Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { ActivityIndicator, type LayoutChangeEvent, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Button } from "@/components/Button";
import { FLOATING_HEADER_HEIGHT, FloatingHeader } from "@/components/FloatingHeader";
import { IconButton } from "@/components/IconButton";
import { createStyles, spacing, type, useColors, useScheme } from "@/lib/theme";

import { washTint } from "../areaColors";
import { primaryArea } from "../helpers";
import { useTemplatePage } from "../useTemplatePage";
import { AreaWash } from "./AreaWash";
import { HireCta } from "./HireCta";
import { TemplateProfile } from "./TemplateProfile";

// How far the wash reaches below the status bar: past the portrait, so the
// fade ends behind the name.
const WASH_HEIGHT = 240;

interface Props {
  templateID: string;
}

export function TemplatePage({ templateID }: Props) {
  const page = useTemplatePage(templateID);
  const insets = useSafeAreaInsets();
  const styles = useStyles();
  const colors = useColors();
  const scheme = useScheme();
  const [ctaHeight, setCtaHeight] = useState(0);
  const headerHeight = insets.top + FLOATING_HEADER_HEIGHT;
  const area = page.template ? primaryArea(page.template) : null;
  const wash = area ? washTint(area, scheme) : null;

  function handleCtaLayout(event: LayoutChangeEvent) {
    setCtaHeight(event.nativeEvent.layout.height);
  }

  function renderBody() {
    if (page.template) {
      return (
        <>
          <ScrollView
            contentContainerStyle={{ paddingTop: headerHeight, paddingBottom: ctaHeight + spacing.sm }}
          >
            <TemplateProfile template={page.template} services={page.services} />
          </ScrollView>
          <HireCta
            name={page.template.name}
            isOnTeam={page.isOnTeam}
            isHiring={page.isHiring}
            isCheckingTeam={page.isCheckingTeam}
            isOpeningChat={page.isOpeningChat}
            error={page.error}
            onHire={page.hire}
            onChat={page.openChat}
            onLayout={handleCtaLayout}
          />
        </>
      );
    }
    if (page.isLoading) {
      return (
        <ActivityIndicator
          style={[styles.loading, { marginTop: headerHeight + spacing.xl }]}
          color={colors.textTertiary}
        />
      );
    }
    if (page.isLoadError) {
      return (
        <View style={[styles.missing, { paddingTop: headerHeight + spacing.xl }]}>
          <Text style={styles.missingText}>Couldn&apos;t load this expert.</Text>
          <Button label="Try again" variant="secondary" onPress={page.retry} />
        </View>
      );
    }
    return (
      <View style={[styles.missing, { paddingTop: headerHeight + spacing.xl }]}>
        <Text style={styles.missingText}>This expert isn&apos;t in the marketplace any more.</Text>
        <Button label="Back to marketplace" variant="secondary" onPress={page.back} />
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      {/* Fixed behind the page, so the header's fade always sits on its colour. */}
      {wash ? (
        <AreaWash color={wash} height={insets.top + WASH_HEIGHT} solidHeight={headerHeight} />
      ) : null}
      {renderBody()}
      <FloatingHeader
        left={
          <IconButton icon={ArrowLeft01Icon} label="Back" onPress={page.back} variant="floating" />
        }
        right={
          page.template ? (
            <IconButton icon={Share08Icon} label="Share" onPress={page.share} variant="floating" />
          ) : null
        }
        fadeColor={wash ?? colors.panelBackground}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1, backgroundColor: colors.panelBackground },
  loading: { alignSelf: "center" },
  missing: { alignItems: "center", gap: spacing.lg, padding: spacing.xl },
  missingText: { ...type.body, color: colors.textSecondary, textAlign: "center" },
}));
