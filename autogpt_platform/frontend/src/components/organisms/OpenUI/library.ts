import { Itinerary } from "@/lib/openui/catalog-itinerary";
import { ItineraryView } from "./components/ItineraryView/ItineraryView";
import { DateTimeField } from "@/lib/openui/catalog-temporal";
import { DateTimeFieldView } from "./components/InputViews/DateTimeFieldView";
import { ExecutionView as ExecutionRenderer } from "./components/ExecutionView";
import { Artifact, ExecutionView } from "@/lib/openui/catalog-artifacts";
import { Dashboard, SeriesChart } from "@/lib/openui/catalog-dashboard";
import { TaskBoard } from "@/lib/openui/catalog-tasks";
import { ArtifactView } from "./components/ArtifactView/ArtifactView";
import { DashboardView } from "./components/DashboardView/DashboardView";
import { SeriesChartView } from "./components/SeriesChartView/SeriesChartView";
import { TaskBoardView } from "./components/TaskBoardView/TaskBoardView";
import { DocumentEditorView } from "./components/DocumentEditorView";
import { RangeFieldView } from "./components/InputViews/RangeFieldView";
import { DateRangeFieldView } from "./components/InputViews/DateRangeFieldView";
import { FieldGroupView } from "./components/InputViews/FieldGroupView";
import { SearchSelectView } from "./components/InputViews/SearchSelectView";
import { ValidatedFieldView } from "./components/InputViews/ValidatedFieldView";
import { RankedListView } from "./components/RankedListView";
import { ResultListView } from "./components/ResultListView/ResultListView";
import { DataGridView } from "./components/DataGridView/DataGridView";
import { DataGrid } from "@/lib/openui/catalog-data";
import { ResultList, RankedList } from "@/lib/openui/catalog-results";
import {
  ValidatedField,
  SearchSelect,
  FieldGroup,
  DateRangeField,
  RangeField,
  DocumentEditor,
} from "@/lib/openui/catalog-inputs";
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
import { Schedule } from "@/lib/openui/catalog-schedule";
import { ScheduleView } from "./components/ScheduleView/ScheduleView";
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
    defineComponent({ ...Itinerary, component: ItineraryView }),
    defineComponent({ ...DateTimeField, component: DateTimeFieldView }),
    defineComponent({ ...ExecutionView, component: ExecutionRenderer }),
    defineComponent({ ...Artifact, component: ArtifactView }),
    defineComponent({ ...Dashboard, component: DashboardView }),
    defineComponent({ ...SeriesChart, component: SeriesChartView }),
    defineComponent({ ...TaskBoard, component: TaskBoardView }),
    defineComponent({ ...DocumentEditor, component: DocumentEditorView }),
    defineComponent({ ...RangeField, component: RangeFieldView }),
    defineComponent({ ...DateRangeField, component: DateRangeFieldView }),
    defineComponent({ ...FieldGroup, component: FieldGroupView }),
    defineComponent({ ...SearchSelect, component: SearchSelectView }),
    defineComponent({ ...ValidatedField, component: ValidatedFieldView }),
    defineComponent({ ...RankedList, component: RankedListView }),
    defineComponent({ ...ResultList, component: ResultListView }),
    defineComponent({ ...DataGrid, component: DataGridView }),
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
    defineComponent({ ...Schedule, component: ScheduleView }),
    defineComponent({ ...CalculatedMetric, component: CalculatedMetricView }),
    defineComponent({ ...TextAreaField, component: TextAreaFieldView }),
    defineComponent({ ...ToggleField, component: ToggleFieldView }),
    defineComponent({ ...MultiSelectField, component: MultiSelectFieldView }),
  ],
});
