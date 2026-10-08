import { createLibrary, defineComponent } from "@openuidev/react-lang";
import * as catalog from "@/lib/openui/catalog";
import {
  ActionView,
  InsightView,
  MetricView,
  MetricsView,
  WorkspaceView,
} from "./components/WorkspaceViews";
import { ChartView } from "./components/ChartView";
import { DataTableView } from "./components/DataTableView/DataTableView";
import { ChecklistView } from "./components/ChecklistView";
import { FieldView, FormView } from "./components/FormViews";
import {
  Map,
  Timeline,
  TrendChart,
  DonutChart,
} from "@/lib/openui/catalog-sections";
import {
  SelectField,
  NumberField,
  DateField,
} from "@/lib/openui/catalog-fields";
import { MapView } from "./components/MapView/MapView";
import { TimelineView } from "./components/TimelineView";
import { TrendChartView } from "./components/TrendChartView";
import { DonutChartView } from "./components/DonutChartView";
import {
  SelectFieldView,
  NumberFieldView,
  DateFieldView,
} from "./components/TypedFieldViews";

export const autoGPTLibrary = createLibrary({
  root: "Workspace",
  components: [
    defineComponent({ ...catalog.Workspace, component: WorkspaceView }),
    defineComponent({ ...catalog.Metric, component: MetricView }),
    defineComponent({ ...catalog.Metrics, component: MetricsView }),
    defineComponent({ ...catalog.Chart, component: ChartView }),
    defineComponent({ ...catalog.DataTable, component: DataTableView }),
    defineComponent({ ...catalog.Insight, component: InsightView }),
    defineComponent({ ...catalog.Checklist, component: ChecklistView }),
    defineComponent({ ...catalog.Field, component: FieldView }),
    defineComponent({ ...catalog.Form, component: FormView }),
    defineComponent({ ...catalog.FollowUp, component: ActionView }),
    defineComponent({ ...Map, component: MapView }),
    defineComponent({ ...Timeline, component: TimelineView }),
    defineComponent({ ...TrendChart, component: TrendChartView }),
    defineComponent({ ...DonutChart, component: DonutChartView }),
    defineComponent({ ...SelectField, component: SelectFieldView }),
    defineComponent({ ...NumberField, component: NumberFieldView }),
    defineComponent({ ...DateField, component: DateFieldView }),
  ],
});
