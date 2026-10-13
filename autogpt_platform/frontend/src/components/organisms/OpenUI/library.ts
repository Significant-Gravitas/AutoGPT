import { createLibrary, defineComponent } from "@openuidev/react-lang";
import * as catalog from "@/lib/openui/catalog";
import { Artifact } from "@/lib/openui/catalog-artifacts";
import {
  DateField,
  Field,
  Form,
  NumberField,
  SelectField,
} from "@/lib/openui/catalog-fields";
import { Itinerary } from "@/lib/openui/catalog-itinerary";
import { Comparison, ResultList } from "@/lib/openui/catalog-results";
import {
  MultiSelectField,
  TextAreaField,
  ToggleField,
} from "@/lib/openui/catalog-rich-fields";
import { Map, SeriesChart, Timeline } from "@/lib/openui/catalog-sections";
import { ArtifactView } from "./components/ArtifactView/ArtifactView";
import { ChecklistView } from "./components/ChecklistView";
import { ComparisonView } from "./components/ComparisonView/ComparisonView";
import { DataTableView } from "./components/DataTableView/DataTableView";
import { FieldView, FormView } from "./components/FormViews";
import { ItineraryView } from "./components/ItineraryView/ItineraryView";
import { MapView } from "./components/MapView/MapView";
import { NumberFieldView } from "./components/NumberFieldView";
import { ProseView } from "./components/ProseView";
import { ResultListView } from "./components/ResultListView/ResultListView";
import {
  MultiSelectFieldView,
  TextAreaFieldView,
  ToggleFieldView,
} from "./components/RichFieldViews";
import { SeriesChartView } from "./components/SeriesChartView/SeriesChartView";
import { TimelineView } from "./components/TimelineView";
import { DateFieldView, SelectFieldView } from "./components/TypedFieldViews";
import {
  ActionView,
  InsightView,
  MetricView,
  MetricsView,
  WorkspaceView,
} from "./components/WorkspaceViews";

export const autoGPTLibrary = createLibrary({
  root: "Workspace",
  components: [
    defineComponent({ ...catalog.Workspace, component: WorkspaceView }),
    defineComponent({ ...catalog.Prose, component: ProseView }),
    defineComponent({ ...catalog.Metrics, component: MetricsView }),
    defineComponent({ ...catalog.Metric, component: MetricView }),
    defineComponent({ ...SeriesChart, component: SeriesChartView }),
    defineComponent({ ...catalog.DataTable, component: DataTableView }),
    defineComponent({ ...catalog.Checklist, component: ChecklistView }),
    defineComponent({ ...Timeline, component: TimelineView }),
    defineComponent({ ...Map, component: MapView }),
    defineComponent({ ...Artifact, component: ArtifactView }),
    defineComponent({ ...catalog.Insight, component: InsightView }),
    defineComponent({ ...Form, component: FormView }),
    defineComponent({ ...catalog.FollowUp, component: ActionView }),
    defineComponent({ ...Itinerary, component: ItineraryView }),
    defineComponent({ ...Comparison, component: ComparisonView }),
    defineComponent({ ...ResultList, component: ResultListView }),
    defineComponent({ ...Field, component: FieldView }),
    defineComponent({ ...TextAreaField, component: TextAreaFieldView }),
    defineComponent({ ...SelectField, component: SelectFieldView }),
    defineComponent({ ...MultiSelectField, component: MultiSelectFieldView }),
    defineComponent({ ...NumberField, component: NumberFieldView }),
    defineComponent({ ...DateField, component: DateFieldView }),
    defineComponent({ ...ToggleField, component: ToggleFieldView }),
  ],
});
