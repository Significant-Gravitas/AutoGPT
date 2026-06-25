export interface NotebookOutput {
  output_type: "stream" | "display_data" | "execute_result" | "error";
  name?: "stdout" | "stderr";
  text?: string | string[];
  data?: Record<string, string | string[]>;
  execution_count?: number | null;
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

export interface NotebookCell {
  cell_type: "code" | "markdown" | "raw";
  source: string | string[];
  outputs?: NotebookOutput[];
  execution_count?: number | null;
  metadata?: Record<string, unknown>;
}

export interface NotebookMetadata {
  kernelspec?: {
    language?: string;
    display_name?: string;
    name?: string;
  };
  language_info?: {
    name?: string;
    version?: string;
  };
}

export interface Notebook {
  nbformat: number;
  nbformat_minor?: number;
  metadata?: NotebookMetadata;
  cells: NotebookCell[];
}
