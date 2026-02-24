import { descriptionId, FieldProps, getTemplate, titleId } from "@rjsf/utils";
import { Table, RowData } from "@/components/molecules/Table/Table";
import { useMemo } from "react";

export const TableField = (props: FieldProps) => {
  const { schema, formData, onChange, fieldPathId, registry, uiSchema } = props;

  const itemSchema = schema.items as any;
  const properties = itemSchema?.properties || {};

  const columns: string[] = useMemo(() => {
    return Object.keys(properties);
  }, [properties]);

  const handleChange = (rows: RowData[]) => {
    onChange(rows, fieldPathId?.path.slice(0, -1));
  };

  const TitleFieldTemplate = getTemplate("TitleFieldTemplate", registry);
  const DescriptionFieldTemplate = getTemplate(
    "DescriptionFieldTemplate",
    registry,
  );

  return (
    <div className="flex flex-col gap-2">
      <TitleFieldTemplate
        id={titleId(fieldPathId)}
        title={schema.title || ""}
        required={true}
        schema={schema}
        uiSchema={uiSchema}
        registry={registry}
      />
      <DescriptionFieldTemplate
        id={descriptionId(fieldPathId)}
        description={schema.description || ""}
        schema={schema}
        registry={registry}
      />

      <Table
        columns={columns}
        defaultValues={formData}
        onChange={handleChange}
        allowAddRow={true}
        allowDeleteRow={true}
        addRowLabel="Add row"
      />
    </div>
  );
};
