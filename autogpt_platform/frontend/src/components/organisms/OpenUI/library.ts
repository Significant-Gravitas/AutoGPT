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
  Field,
  Form,
  SelectField,
  NumberField,
  DateField,
} from "@/lib/openui/catalog-fields";
import { MapView } from "./components/MapView/MapView";
import { TimelineView } from "./components/TimelineView";
import { TrendChartView } from "./components/TrendChartView";
import { DonutChartView } from "./components/DonutChartView";
import { SelectFieldView, DateFieldView } from "./components/TypedFieldViews";
import { NumberFieldView } from "./components/NumberFieldView";
import {
  Comparison,
  CostTable,
  CalculatedMetric,
} from "@/lib/openui/catalog-connected";
import {
  TextAreaField,
  ToggleField,
  MultiSelectField,
} from "@/lib/openui/catalog-rich-fields";
import { ComparisonView } from "./components/ComparisonView/ComparisonView";
import { CostTableView } from "./components/CostTableView/CostTableView";
import { CalculatedMetricView } from "./components/CalculatedMetricView";
import {
  TextAreaFieldView,
  ToggleFieldView,
  MultiSelectFieldView,
} from "./components/RichFieldViews";

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
    defineComponent({ ...Field, component: FieldView }),
    defineComponent({ ...Form, component: FormView }),
    defineComponent({ ...catalog.FollowUp, component: ActionView }),
    defineComponent({ ...Map, component: MapView }),
    defineComponent({ ...Timeline, component: TimelineView }),
    defineComponent({ ...TrendChart, component: TrendChartView }),
    defineComponent({ ...DonutChart, component: DonutChartView }),
    defineComponent({ ...SelectField, component: SelectFieldView }),
    defineComponent({ ...NumberField, component: NumberFieldView }),
    defineComponent({ ...DateField, component: DateFieldView }),
    defineComponent({ ...Comparison, component: ComparisonView }),
    defineComponent({ ...CostTable, component: CostTableView }),
    defineComponent({ ...CalculatedMetric, component: CalculatedMetricView }),
    defineComponent({ ...TextAreaField, component: TextAreaFieldView }),
    defineComponent({ ...ToggleField, component: ToggleFieldView }),
    defineComponent({ ...MultiSelectField, component: MultiSelectFieldView }),
  ],
});
