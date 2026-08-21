"use client";

import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";
import { ToolUIPreview } from "./components/ToolUIPreview/ToolUIPreview";

export default withFeatureFlag(ToolUIPreview, "new-tool-ui");
