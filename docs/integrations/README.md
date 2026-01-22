# AutoGPT Blocks Overview

AutoGPT uses a modular approach with various "blocks" to handle different tasks. These blocks are the building blocks of AutoGPT workflows, allowing users to create complex automations by combining simple, specialized components.

!!! info "Creating Your Own Blocks"
    Want to create your own custom blocks? Check out our guides:
    
    - [Build your own Blocks](https://docs.agpt.co/platform/new_blocks/) - Step-by-step tutorial with examples
    - [Block SDK Guide](https://docs.agpt.co/platform/block-sdk-guide/) - Advanced SDK patterns with OAuth, webhooks, and provider configuration

Below is a comprehensive list of all available blocks, categorized by their primary function. Click on any block name to view its detailed documentation.

## Basic Operations

| Block Name | Description |
|------------|-------------|
| [Add Memory](basic.md#add-memory) | Add new memories to Mem0 with user segmentation |
| [Add To Dictionary](basic.md#add-to-dictionary) | Adds a new key-value pair to a dictionary |
| [Add To Library From Store](system/library_operations.md#add-to-library-from-store) | Add an agent from the store to your personal library |
| [Add To List](basic.md#add-to-list) | Adds a new entry to a list |
| [Agent Date Input](basic.md#agent-date-input) | Block for date input |
| [Agent Dropdown Input](basic.md#agent-dropdown-input) | Block for dropdown text selection |
| [Agent File Input](basic.md#agent-file-input) | Block for file upload input (string path for example) |
| [Agent Google Drive File Input](basic.md#agent-google-drive-file-input) | Block for selecting a file from Google Drive |
| [Agent Input](basic.md#agent-input) | A block that accepts and processes user input values within a workflow, supporting various input types and validation |
| [Agent Long Text Input](basic.md#agent-long-text-input) | Block for long text input (multi-line) |
| [Agent Number Input](basic.md#agent-number-input) | Block for number input |
| [Agent Output](basic.md#agent-output) | A block that records and formats workflow results for display to users, with optional Jinja2 template formatting support |
| [Agent Short Text Input](basic.md#agent-short-text-input) | Block for short text input (single-line) |
| [Agent Table Input](basic.md#agent-table-input) | Block for table data input with customizable headers |
| [Agent Time Input](basic.md#agent-time-input) | Block for time input |
| [Agent Toggle Input](basic.md#agent-toggle-input) | Block for boolean toggle input |
| [Block Installation](basic.md#block-installation) | Given a code string, this block allows the verification and installation of a block code into the system |
| [Concatenate Lists](basic.md#concatenate-lists) | Concatenates multiple lists into a single list |
| [Dictionary Is Empty](basic.md#dictionary-is-empty) | Checks if a dictionary is empty |
| [File Store](basic.md#file-store) | Stores the input file in the temporary directory |
| [Find In Dictionary](basic.md#find-in-dictionary) | A block that looks up a value in a dictionary, list, or object by key or index and returns the corresponding value |
| [Find In List](basic.md#find-in-list) | Finds the index of the value in the list |
| [Get All Memories](basic.md#get-all-memories) | Retrieve all memories from Mem0 with optional conversation filtering |
| [Get Latest Memory](basic.md#get-latest-memory) | Retrieve the latest memory from Mem0 with optional key filtering |
| [Get List Item](basic.md#get-list-item) | Returns the element at the given index |
| [Get Store Agent Details](system/store_operations.md#get-store-agent-details) | Get detailed information about an agent from the store |
| [Get Weather Information](basic.md#get-weather-information) | Retrieves weather information for a specified location using OpenWeatherMap API |
| [Human In The Loop](basic.md#human-in-the-loop) | Pause execution and wait for human approval or modification of data |
| [Linear Search Issues](linear/issues.md#linear-search-issues) | Searches for issues on Linear |
| [List Is Empty](basic.md#list-is-empty) | Checks if a list is empty |
| [List Library Agents](system/library_operations.md#list-library-agents) | List all agents in your personal library |
| [Note](basic.md#note) | A visual annotation block that displays a sticky note in the workflow editor for documentation and organization purposes |
| [Print To Console](basic.md#print-to-console) | A debugging block that outputs text to the console for monitoring and troubleshooting workflow execution |
| [Remove From Dictionary](basic.md#remove-from-dictionary) | Removes a key-value pair from a dictionary |
| [Remove From List](basic.md#remove-from-list) | Removes an item from a list by value or index |
| [Replace Dictionary Value](basic.md#replace-dictionary-value) | Replaces the value for a specified key in a dictionary |
| [Replace List Item](basic.md#replace-list-item) | Replaces an item at the specified index |
| [Reverse List Order](basic.md#reverse-list-order) | Reverses the order of elements in a list |
| [Search Memory](basic.md#search-memory) | Search memories in Mem0 by user |
| [Search Store Agents](system/store_operations.md#search-store-agents) | Search for agents in the store |
| [Slant3D Cancel Order](slant3d/order.md#slant3d-cancel-order) | Cancel an existing order |
| [Slant3D Create Order](slant3d/order.md#slant3d-create-order) | Create a new print order |
| [Slant3D Estimate Order](slant3d/order.md#slant3d-estimate-order) | Get order cost estimate |
| [Slant3D Estimate Shipping](slant3d/order.md#slant3d-estimate-shipping) | Get shipping cost estimate |
| [Slant3D Filament](slant3d/filament.md#slant3d-filament) | Get list of available filaments |
| [Slant3D Get Orders](slant3d/order.md#slant3d-get-orders) | Get all orders for the account |
| [Slant3D Slicer](slant3d/slicing.md#slant3d-slicer) | Slice a 3D model file and get pricing information |
| [Slant3D Tracking](slant3d/order.md#slant3d-tracking) | Track order status and shipping |
| [Store Value](basic.md#store-value) | A basic block that stores and forwards a value throughout workflows, allowing it to be reused without changes across multiple blocks |
| [Universal Type Converter](basic.md#universal-type-converter) | This block is used to convert a value to a universal type |
| [XML Parser](basic.md#xml-parser) | Parses XML using gravitasml to tokenize and coverts it to dict |

## Data Processing

| Block Name | Description |
|------------|-------------|
| [Airtable Create Base](airtable/bases.md#airtable-create-base) | Create or find a base in Airtable |
| [Airtable Create Field](airtable/schema.md#airtable-create-field) | Add a new field to an Airtable table |
| [Airtable Create Records](airtable/records.md#airtable-create-records) | Create records in an Airtable table |
| [Airtable Create Table](airtable/schema.md#airtable-create-table) | Create a new table in an Airtable base |
| [Airtable Delete Records](airtable/records.md#airtable-delete-records) | Delete records from an Airtable table |
| [Airtable Get Record](airtable/records.md#airtable-get-record) | Get a single record from Airtable |
| [Airtable List Bases](airtable/bases.md#airtable-list-bases) | List all bases in Airtable |
| [Airtable List Records](airtable/records.md#airtable-list-records) | List records from an Airtable table |
| [Airtable List Schema](airtable/schema.md#airtable-list-schema) | Get the complete schema of an Airtable base |
| [Airtable Update Field](airtable/schema.md#airtable-update-field) | Update field properties in an Airtable table |
| [Airtable Update Records](airtable/records.md#airtable-update-records) | Update records in an Airtable table |
| [Airtable Update Table](airtable/schema.md#airtable-update-table) | Update table properties |
| [Airtable Webhook Trigger](airtable/triggers.md#airtable-webhook-trigger) | Starts a flow whenever Airtable emits a webhook event |
| [Baas Bot Delete Recording](baas/bots.md#baas-bot-delete-recording) | Permanently delete a meeting's recorded data |
| [Baas Bot Fetch Meeting Data](baas/bots.md#baas-bot-fetch-meeting-data) | Retrieve recorded meeting data |
| [Create Dictionary](data.md#create-dictionary) | Creates a dictionary with the specified key-value pairs |
| [Create List](data.md#create-list) | Creates a list with the specified values |
| [Data For Seo Keyword Suggestions](dataforseo/keyword_suggestions.md#data-for-seo-keyword-suggestions) | Get keyword suggestions from DataForSEO Labs Google API |
| [Data For Seo Related Keywords](dataforseo/related_keywords.md#data-for-seo-related-keywords) | Get related keywords from DataForSEO Labs Google API |
| [Exa Create Import](exa/websets_import_export.md#exa-create-import) | Import CSV data to use with websets for targeted searches |
| [Exa Delete Import](exa/websets_import_export.md#exa-delete-import) | Delete an import |
| [Exa Export Webset](exa/websets_import_export.md#exa-export-webset) | Export webset data in JSON, CSV, or JSON Lines format |
| [Exa Get Import](exa/websets_import_export.md#exa-get-import) | Get the status and details of an import |
| [Exa Get New Items](exa/websets_items.md#exa-get-new-items) | Get items added since a cursor - enables incremental processing without reprocessing |
| [Exa List Imports](exa/websets_import_export.md#exa-list-imports) | List all imports with pagination support |
| [File Read](data.md#file-read) | Reads a file and returns its content as a string, with optional chunking by delimiter and size limits |
| [Google Calendar Read Events](google/calendar.md#google-calendar-read-events) | Retrieves upcoming events from a Google Calendar with filtering options |
| [Google Docs Append Markdown](google/docs.md#google-docs-append-markdown) | Append Markdown content to the end of a Google Doc with full formatting - ideal for LLM/AI output |
| [Google Docs Append Plain Text](google/docs.md#google-docs-append-plain-text) | Append plain text to the end of a Google Doc (no formatting applied) |
| [Google Docs Create](google/docs.md#google-docs-create) | Create a new Google Doc |
| [Google Docs Delete Content](google/docs.md#google-docs-delete-content) | Delete a range of content from a Google Doc |
| [Google Docs Export](google/docs.md#google-docs-export) | Export a Google Doc to PDF, Word, text, or other formats |
| [Google Docs Find Replace Plain Text](google/docs.md#google-docs-find-replace-plain-text) | Find and replace plain text in a Google Doc (no formatting applied to replacement) |
| [Google Docs Format Text](google/docs.md#google-docs-format-text) | Apply formatting (bold, italic, color, etc |
| [Google Docs Get Metadata](google/docs.md#google-docs-get-metadata) | Get metadata about a Google Doc |
| [Google Docs Get Structure](google/docs.md#google-docs-get-structure) | Get document structure with index positions for precise editing operations |
| [Google Docs Insert Markdown At](google/docs.md#google-docs-insert-markdown-at) | Insert formatted Markdown at a specific position in a Google Doc - ideal for LLM/AI output |
| [Google Docs Insert Page Break](google/docs.md#google-docs-insert-page-break) | Insert a page break into a Google Doc |
| [Google Docs Insert Plain Text](google/docs.md#google-docs-insert-plain-text) | Insert plain text at a specific position in a Google Doc (no formatting applied) |
| [Google Docs Insert Table](google/docs.md#google-docs-insert-table) | Insert a table into a Google Doc, optionally with content and Markdown formatting |
| [Google Docs Read](google/docs.md#google-docs-read) | Read text content from a Google Doc |
| [Google Docs Replace All With Markdown](google/docs.md#google-docs-replace-all-with-markdown) | Replace entire Google Doc content with formatted Markdown - ideal for LLM/AI output |
| [Google Docs Replace Content With Markdown](google/docs.md#google-docs-replace-content-with-markdown) | Find text and replace it with formatted Markdown - ideal for LLM/AI output and templates |
| [Google Docs Replace Range With Markdown](google/docs.md#google-docs-replace-range-with-markdown) | Replace a specific index range in a Google Doc with formatted Markdown - ideal for LLM/AI output |
| [Google Docs Set Public Access](google/docs.md#google-docs-set-public-access) | Make a Google Doc public or private |
| [Google Docs Share](google/docs.md#google-docs-share) | Share a Google Doc with specific users |
| [Google Sheets Add Column](google/sheets.md#google-sheets-add-column) | Add a new column with a header |
| [Google Sheets Add Dropdown](google/sheets.md#google-sheets-add-dropdown) | Add a dropdown list (data validation) to cells |
| [Google Sheets Add Note](google/sheets.md#google-sheets-add-note) | Add a note to a cell in a Google Sheet |
| [Google Sheets Append Row](google/sheets.md#google-sheets-append-row) | Append or Add a single row to the end of a Google Sheet |
| [Google Sheets Batch Operations](google/sheets.md#google-sheets-batch-operations) | This block performs multiple operations on a Google Sheets spreadsheet in a single batch request |
| [Google Sheets Clear](google/sheets.md#google-sheets-clear) | This block clears data from a specified range in a Google Sheets spreadsheet |
| [Google Sheets Copy To Spreadsheet](google/sheets.md#google-sheets-copy-to-spreadsheet) | Copy a sheet from one spreadsheet to another |
| [Google Sheets Create Named Range](google/sheets.md#google-sheets-create-named-range) | Create a named range to reference cells by name instead of A1 notation |
| [Google Sheets Create Spreadsheet](google/sheets.md#google-sheets-create-spreadsheet) | This block creates a new Google Sheets spreadsheet with specified sheets |
| [Google Sheets Delete Column](google/sheets.md#google-sheets-delete-column) | Delete a column by header name or column letter |
| [Google Sheets Delete Rows](google/sheets.md#google-sheets-delete-rows) | Delete specific rows from a Google Sheet by their row indices |
| [Google Sheets Export Csv](google/sheets.md#google-sheets-export-csv) | Export a Google Sheet as CSV data |
| [Google Sheets Filter Rows](google/sheets.md#google-sheets-filter-rows) | Filter rows in a Google Sheet based on a column condition |
| [Google Sheets Find](google/sheets.md#google-sheets-find) | Find text in a Google Sheets spreadsheet |
| [Google Sheets Find Replace](google/sheets.md#google-sheets-find-replace) | This block finds and replaces text in a Google Sheets spreadsheet |
| [Google Sheets Format](google/sheets.md#google-sheets-format) | Format a range in a Google Sheet (sheet optional) |
| [Google Sheets Get Column](google/sheets.md#google-sheets-get-column) | Extract all values from a specific column |
| [Google Sheets Get Notes](google/sheets.md#google-sheets-get-notes) | Get notes from cells in a Google Sheet |
| [Google Sheets Get Row](google/sheets.md#google-sheets-get-row) | Get a specific row by its index |
| [Google Sheets Get Row Count](google/sheets.md#google-sheets-get-row-count) | Get row count and dimensions of a Google Sheet |
| [Google Sheets Get Unique Values](google/sheets.md#google-sheets-get-unique-values) | Get unique values from a column |
| [Google Sheets Import Csv](google/sheets.md#google-sheets-import-csv) | Import CSV data into a Google Sheet |
| [Google Sheets Insert Row](google/sheets.md#google-sheets-insert-row) | Insert a single row at a specific position |
| [Google Sheets List Named Ranges](google/sheets.md#google-sheets-list-named-ranges) | List all named ranges in a spreadsheet |
| [Google Sheets Lookup Row](google/sheets.md#google-sheets-lookup-row) | Look up a row by finding a value in a specific column |
| [Google Sheets Manage Sheet](google/sheets.md#google-sheets-manage-sheet) | Create, delete, or copy sheets (sheet optional) |
| [Google Sheets Metadata](google/sheets.md#google-sheets-metadata) | This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties |
| [Google Sheets Protect Range](google/sheets.md#google-sheets-protect-range) | Protect a cell range or entire sheet from editing |
| [Google Sheets Read](google/sheets.md#google-sheets-read) | A block that reads data from a Google Sheets spreadsheet using A1 notation range selection |
| [Google Sheets Remove Duplicates](google/sheets.md#google-sheets-remove-duplicates) | Remove duplicate rows based on specified columns |
| [Google Sheets Set Public Access](google/sheets.md#google-sheets-set-public-access) | Make a Google Spreadsheet public or private |
| [Google Sheets Share Spreadsheet](google/sheets.md#google-sheets-share-spreadsheet) | Share a Google Spreadsheet with users or get shareable link |
| [Google Sheets Sort](google/sheets.md#google-sheets-sort) | Sort a Google Sheet by one or two columns |
| [Google Sheets Update Cell](google/sheets.md#google-sheets-update-cell) | Update a single cell in a Google Sheets spreadsheet |
| [Google Sheets Update Row](google/sheets.md#google-sheets-update-row) | Update a specific row by its index |
| [Google Sheets Write](google/sheets.md#google-sheets-write) | A block that writes data to a Google Sheets spreadsheet at a specified A1 notation range |
| [Keyword Suggestion Extractor](dataforseo/keyword_suggestions.md#keyword-suggestion-extractor) | Extract individual fields from a KeywordSuggestion object |
| [Persist Information](data.md#persist-information) | Persist key-value information for the current user |
| [Read Spreadsheet](data.md#read-spreadsheet) | Reads CSV and Excel files and outputs the data as a list of dictionaries and individual rows |
| [Related Keyword Extractor](dataforseo/related_keywords.md#related-keyword-extractor) | Extract individual fields from a RelatedKeyword object |
| [Retrieve Information](data.md#retrieve-information) | Retrieve key-value information for the current user |
| [Screenshot Web Page](data.md#screenshot-web-page) | Takes a screenshot of a specified website using ScreenshotOne API |

## Text Processing

| Block Name | Description |
|------------|-------------|
| [Code Extraction](text.md#code-extraction) | Extracts code blocks from text and identifies their programming languages |
| [Combine Texts](text.md#combine-texts) | This block combines multiple input texts into a single output text |
| [Countdown Timer](text.md#countdown-timer) | This block triggers after a specified duration |
| [Extract Text Information](text.md#extract-text-information) | This block extracts the text from the given text using the pattern (regex) |
| [Fill Text Template](text.md#fill-text-template) | This block formats the given texts using the format template |
| [Get Current Date](text.md#get-current-date) | This block outputs the current date with an optional offset |
| [Get Current Date And Time](text.md#get-current-date-and-time) | This block outputs the current date and time |
| [Get Current Time](text.md#get-current-time) | This block outputs the current time |
| [Match Text Pattern](text.md#match-text-pattern) | Matches text against a regex pattern and forwards data to positive or negative output based on the match |
| [Text Decoder](text.md#text-decoder) | Decodes a string containing escape sequences into actual text |
| [Text Replace](text.md#text-replace) | This block is used to replace a text with a new text |
| [Text Split](text.md#text-split) | This block is used to split a text into a list of strings |
| [Word Character Count](text.md#word-character-count) | Counts the number of words and characters in a given text |

## AI and Language Models

| Block Name | Description |
|------------|-------------|
| [AI Ad Maker Video Creator](llm.md#ai-ad-maker-video-creator) | Creates an AI‑generated 30‑second advert (text + images) |
| [AI Condition](llm.md#ai-condition) | Uses AI to evaluate natural language conditions and provide conditional outputs |
| [AI Conversation](llm.md#ai-conversation) | A block that facilitates multi-turn conversations with a Large Language Model (LLM), maintaining context across message exchanges |
| [AI Image Customizer](llm.md#ai-image-customizer) | Generate and edit custom images using Google's Nano-Banana model from Gemini 2 |
| [AI Image Editor](llm.md#ai-image-editor) | Edit images using BlackForest Labs' Flux Kontext models |
| [AI Image Generator](llm.md#ai-image-generator) | Generate images using various AI models through a unified interface |
| [AI List Generator](llm.md#ai-list-generator) | A block that creates lists of items based on prompts using a Large Language Model (LLM), with optional source data for context |
| [AI Music Generator](llm.md#ai-music-generator) | This block generates music using Meta's MusicGen model on Replicate |
| [AI Screenshot To Video Ad](llm.md#ai-screenshot-to-video-ad) | Turns a screenshot into an engaging, avatar‑narrated video advert |
| [AI Shortform Video Creator](llm.md#ai-shortform-video-creator) | Creates a shortform video using revid |
| [AI Structured Response Generator](llm.md#ai-structured-response-generator) | A block that generates structured JSON responses using a Large Language Model (LLM), with schema validation and format enforcement |
| [AI Text Generator](llm.md#ai-text-generator) | A block that produces text responses using a Large Language Model (LLM) based on customizable prompts and system instructions |
| [AI Text Summarizer](llm.md#ai-text-summarizer) | A block that summarizes long texts using a Large Language Model (LLM), with configurable focus topics and summary styles |
| [AI Video Generator](fal/ai_video_generator.md#ai-video-generator) | Generate videos using FAL AI models |
| [Bannerbear Text Overlay](bannerbear/text_overlay.md#bannerbear-text-overlay) | Add text overlay to images using Bannerbear templates |
| [Code Generation](llm.md#code-generation) | Generate or refactor code using OpenAI's Codex (Responses API) |
| [Create Talking Avatar Video](llm.md#create-talking-avatar-video) | This block integrates with D-ID to create video clips and retrieve their URLs |
| [Exa Answer](exa/answers.md#exa-answer) | Get an LLM answer to a question informed by Exa search results |
| [Exa Create Enrichment](exa/websets_enrichment.md#exa-create-enrichment) | Create enrichments to extract additional structured data from webset items |
| [Exa Create Research](exa/research.md#exa-create-research) | Create research task with optional waiting - explores web and synthesizes findings with citations |
| [Ideogram Model](llm.md#ideogram-model) | This block runs Ideogram models with both simple and advanced settings |
| [Jina Chunking](jina/chunking.md#jina-chunking) | Chunks texts using Jina AI's segmentation service |
| [Jina Embedding](jina/embeddings.md#jina-embedding) | Generates embeddings using Jina AI |
| [Perplexity](llm.md#perplexity) | Query Perplexity's sonar models with real-time web search capabilities and receive annotated responses with source citations |
| [Replicate Flux Advanced Model](replicate/flux_advanced.md#replicate-flux-advanced-model) | This block runs Flux models on Replicate with advanced settings |
| [Replicate Model](replicate/replicate_block.md#replicate-model) | Run Replicate models synchronously |
| [Smart Decision Maker](llm.md#smart-decision-maker) | Uses AI to intelligently decide what tool to use |
| [Stagehand Act](stagehand/blocks.md#stagehand-act) | Interact with a web page by performing actions on a web page |
| [Stagehand Extract](stagehand/blocks.md#stagehand-extract) | Extract structured data from a webpage |
| [Stagehand Observe](stagehand/blocks.md#stagehand-observe) | Find suggested actions for your workflows |
| [Unreal Text To Speech](llm.md#unreal-text-to-speech) | Converts text to speech using the Unreal Speech API |

## Search and Information Retrieval

| Block Name | Description |
|------------|-------------|
| [Ask Wolfram](wolfram/llm_api.md#ask-wolfram) | Ask Wolfram Alpha a question |
| [Exa Bulk Webset Items](exa/websets_items.md#exa-bulk-webset-items) | Get all items from a webset in bulk (with configurable limits) |
| [Exa Cancel Enrichment](exa/websets_enrichment.md#exa-cancel-enrichment) | Cancel a running enrichment operation |
| [Exa Cancel Webset](exa/websets.md#exa-cancel-webset) | Cancel all operations being performed on a Webset |
| [Exa Cancel Webset Search](exa/websets_search.md#exa-cancel-webset-search) | Cancel a running webset search |
| [Exa Contents](exa/contents.md#exa-contents) | Retrieves document contents using Exa's contents API |
| [Exa Create Monitor](exa/websets_monitor.md#exa-create-monitor) | Create automated monitors to keep websets updated with fresh data on a schedule |
| [Exa Create Or Find Webset](exa/websets.md#exa-create-or-find-webset) | Create a new webset or return existing one by external_id (idempotent operation) |
| [Exa Create Webset](exa/websets.md#exa-create-webset) | Create a new Exa Webset for persistent web search collections with optional waiting for initial results |
| [Exa Create Webset Search](exa/websets_search.md#exa-create-webset-search) | Add a new search to an existing webset to find more items |
| [Exa Delete Enrichment](exa/websets_enrichment.md#exa-delete-enrichment) | Delete an enrichment from a webset |
| [Exa Delete Monitor](exa/websets_monitor.md#exa-delete-monitor) | Delete a monitor from a webset |
| [Exa Delete Webset](exa/websets.md#exa-delete-webset) | Delete a Webset and all its items |
| [Exa Delete Webset Item](exa/websets_items.md#exa-delete-webset-item) | Delete a specific item from a webset |
| [Exa Find Or Create Search](exa/websets_search.md#exa-find-or-create-search) | Find existing search by query or create new - prevents duplicate searches in workflows |
| [Exa Find Similar](exa/similar.md#exa-find-similar) | Finds similar links using Exa's findSimilar API |
| [Exa Get Enrichment](exa/websets_enrichment.md#exa-get-enrichment) | Get the status and details of a webset enrichment |
| [Exa Get Monitor](exa/websets_monitor.md#exa-get-monitor) | Get the details and status of a webset monitor |
| [Exa Get Research](exa/research.md#exa-get-research) | Get status and results of a research task |
| [Exa Get Webset](exa/websets.md#exa-get-webset) | Retrieve a Webset by ID or external ID |
| [Exa Get Webset Item](exa/websets_items.md#exa-get-webset-item) | Get a specific item from a webset by its ID |
| [Exa Get Webset Search](exa/websets_search.md#exa-get-webset-search) | Get the status and details of a webset search |
| [Exa List Monitors](exa/websets_monitor.md#exa-list-monitors) | List all monitors with optional webset filtering |
| [Exa List Research](exa/research.md#exa-list-research) | List all research tasks with pagination support |
| [Exa List Webset Items](exa/websets_items.md#exa-list-webset-items) | List items in a webset with pagination support |
| [Exa List Websets](exa/websets.md#exa-list-websets) | List all Websets with pagination support |
| [Exa Preview Webset](exa/websets.md#exa-preview-webset) | Preview how a search query will be interpreted before creating a webset |
| [Exa Search](exa/search.md#exa-search) | Searches the web using Exa's advanced search API |
| [Exa Update Enrichment](exa/websets_enrichment.md#exa-update-enrichment) | Update an existing enrichment configuration |
| [Exa Update Monitor](exa/websets_monitor.md#exa-update-monitor) | Update a monitor's status, schedule, or metadata |
| [Exa Update Webset](exa/websets.md#exa-update-webset) | Update metadata for an existing Webset |
| [Exa Wait For Enrichment](exa/websets_polling.md#exa-wait-for-enrichment) | Wait for a webset enrichment to complete with progress tracking |
| [Exa Wait For Research](exa/research.md#exa-wait-for-research) | Wait for a research task to complete with configurable timeout |
| [Exa Wait For Search](exa/websets_polling.md#exa-wait-for-search) | Wait for a specific webset search to complete with progress tracking |
| [Exa Wait For Webset](exa/websets_polling.md#exa-wait-for-webset) | Wait for a webset to reach a specific status with progress tracking |
| [Exa Webset Items Summary](exa/websets_items.md#exa-webset-items-summary) | Get a summary of webset items without retrieving all data |
| [Exa Webset Status](exa/websets.md#exa-webset-status) | Get a quick status overview of a webset |
| [Exa Webset Summary](exa/websets.md#exa-webset-summary) | Get a comprehensive summary of a webset with samples and statistics |
| [Extract Website Content](jina/search.md#extract-website-content) | This block scrapes the content from the given web URL |
| [Fact Checker](jina/fact_checker.md#fact-checker) | This block checks the factuality of a given statement using Jina AI's Grounding API |
| [Firecrawl Crawl](firecrawl/crawl.md#firecrawl-crawl) | Firecrawl crawls websites to extract comprehensive data while bypassing blockers |
| [Firecrawl Extract](firecrawl/extract.md#firecrawl-extract) | Firecrawl crawls websites to extract comprehensive data while bypassing blockers |
| [Firecrawl Map Website](firecrawl/map.md#firecrawl-map-website) | Firecrawl maps a website to extract all the links |
| [Firecrawl Scrape](firecrawl/scrape.md#firecrawl-scrape) | Firecrawl scrapes a website to extract comprehensive data while bypassing blockers |
| [Firecrawl Search](firecrawl/search.md#firecrawl-search) | Firecrawl searches the web for the given query |
| [Get Person Detail](apollo/person.md#get-person-detail) | Get detailed person data with Apollo API, including email reveal |
| [Get Wikipedia Summary](search.md#get-wikipedia-summary) | This block fetches the summary of a given topic from Wikipedia |
| [Google Maps Search](search.md#google-maps-search) | This block searches for local businesses using Google Maps API |
| [Search Organizations](apollo/organization.md#search-organizations) | Search for organizations in Apollo |
| [Search People](apollo/people.md#search-people) | Search for people in Apollo |
| [Search The Web](jina/search.md#search-the-web) | This block searches the internet for the given search query |
| [Validate Emails](zerobounce/validate_emails.md#validate-emails) | Validate emails |

## Social Media and Content

| Block Name | Description |
|------------|-------------|
| [Create Discord Thread](discord/bot_blocks.md#create-discord-thread) | Creates a new thread in a Discord channel |
| [Create Reddit Post](misc.md#create-reddit-post) | Create a new post on a subreddit |
| [Delete Reddit Comment](misc.md#delete-reddit-comment) | Delete a Reddit comment that you own |
| [Delete Reddit Post](misc.md#delete-reddit-post) | Delete a Reddit post that you own |
| [Discord Channel Info](discord/bot_blocks.md#discord-channel-info) | Resolves Discord channel names to IDs and vice versa |
| [Discord Get Current User](discord/oauth_blocks.md#discord-get-current-user) | Gets information about the currently authenticated Discord user using OAuth2 credentials |
| [Discord User Info](discord/bot_blocks.md#discord-user-info) | Gets information about a Discord user by their ID |
| [Edit Reddit Post](misc.md#edit-reddit-post) | Edit the body text of an existing Reddit post that you own |
| [Get Linkedin Profile](enrichlayer/linkedin.md#get-linkedin-profile) | Fetch LinkedIn profile data using Enrichlayer |
| [Get Linkedin Profile Picture](enrichlayer/linkedin.md#get-linkedin-profile-picture) | Get LinkedIn profile pictures using Enrichlayer |
| [Get Reddit Comment](misc.md#get-reddit-comment) | Get details about a specific Reddit comment by its ID |
| [Get Reddit Comment Replies](misc.md#get-reddit-comment-replies) | Get replies to a specific Reddit comment |
| [Get Reddit Inbox](misc.md#get-reddit-inbox) | Get messages, mentions, and comment replies from your Reddit inbox |
| [Get Reddit Post](misc.md#get-reddit-post) | Get detailed information about a specific Reddit post by its ID |
| [Get Reddit Post Comments](misc.md#get-reddit-post-comments) | Get top-level comments on a Reddit post |
| [Get Reddit Posts](misc.md#get-reddit-posts) | This block fetches Reddit posts from a defined subreddit name |
| [Get Reddit User Info](misc.md#get-reddit-user-info) | Get information about a Reddit user including karma, account age, and verification status |
| [Get Subreddit Flairs](misc.md#get-subreddit-flairs) | Get available link flair options for a subreddit |
| [Get Subreddit Info](misc.md#get-subreddit-info) | Get information about a subreddit including subscriber count, description, and rules |
| [Get Subreddit Rules](misc.md#get-subreddit-rules) | Get the rules for a subreddit to ensure compliance before posting |
| [Get User Posts](misc.md#get-user-posts) | Fetch posts by a specific Reddit user |
| [Linkedin Person Lookup](enrichlayer/linkedin.md#linkedin-person-lookup) | Look up LinkedIn profiles by person information using Enrichlayer |
| [Linkedin Role Lookup](enrichlayer/linkedin.md#linkedin-role-lookup) | Look up LinkedIn profiles by role in a company using Enrichlayer |
| [Post Reddit Comment](misc.md#post-reddit-comment) | This block posts a Reddit comment on a specified Reddit post |
| [Post To Bluesky](ayrshare/post_to_bluesky.md#post-to-bluesky) | Post to Bluesky using Ayrshare |
| [Post To Facebook](ayrshare/post_to_facebook.md#post-to-facebook) | Post to Facebook using Ayrshare |
| [Post To GMB](ayrshare/post_to_gmb.md#post-to-gmb) | Post to Google My Business using Ayrshare |
| [Post To Instagram](ayrshare/post_to_instagram.md#post-to-instagram) | Post to Instagram using Ayrshare |
| [Post To Linked In](ayrshare/post_to_linkedin.md#post-to-linked-in) | Post to LinkedIn using Ayrshare |
| [Post To Pinterest](ayrshare/post_to_pinterest.md#post-to-pinterest) | Post to Pinterest using Ayrshare |
| [Post To Reddit](ayrshare/post_to_reddit.md#post-to-reddit) | Post to Reddit using Ayrshare |
| [Post To Snapchat](ayrshare/post_to_snapchat.md#post-to-snapchat) | Post to Snapchat using Ayrshare |
| [Post To Telegram](ayrshare/post_to_telegram.md#post-to-telegram) | Post to Telegram using Ayrshare |
| [Post To Threads](ayrshare/post_to_threads.md#post-to-threads) | Post to Threads using Ayrshare |
| [Post To Tik Tok](ayrshare/post_to_tiktok.md#post-to-tik-tok) | Post to TikTok using Ayrshare |
| [Post To X](ayrshare/post_to_x.md#post-to-x) | Post to X / Twitter using Ayrshare |
| [Post To You Tube](ayrshare/post_to_youtube.md#post-to-you-tube) | Post to YouTube using Ayrshare |
| [Publish To Medium](misc.md#publish-to-medium) | Publishes a post to Medium |
| [Read Discord Messages](discord/bot_blocks.md#read-discord-messages) | Reads messages from a Discord channel using a bot token |
| [Reddit Get My Posts](misc.md#reddit-get-my-posts) | Fetch posts created by the authenticated Reddit user (you) |
| [Reply To Discord Message](discord/bot_blocks.md#reply-to-discord-message) | Replies to a specific Discord message |
| [Reply To Reddit Comment](misc.md#reply-to-reddit-comment) | Reply to a specific Reddit comment |
| [Search Reddit](misc.md#search-reddit) | Search Reddit for posts matching a query |
| [Send Discord DM](discord/bot_blocks.md#send-discord-dm) | Sends a direct message to a Discord user using their user ID |
| [Send Discord Embed](discord/bot_blocks.md#send-discord-embed) | Sends a rich embed message to a Discord channel |
| [Send Discord File](discord/bot_blocks.md#send-discord-file) | Sends a file attachment to a Discord channel |
| [Send Discord Message](discord/bot_blocks.md#send-discord-message) | Sends a message to a Discord channel using a bot token |
| [Send Reddit Message](misc.md#send-reddit-message) | Send a private message (DM) to a Reddit user |
| [Transcribe Youtube Video](misc.md#transcribe-youtube-video) | Transcribes a YouTube video using a proxy |
| [Twitter Add List Member](twitter/list_members.md#twitter-add-list-member) | This block adds a specified user to a Twitter List owned by the authenticated user |
| [Twitter Bookmark Tweet](twitter/bookmark.md#twitter-bookmark-tweet) | This block bookmarks a tweet on Twitter |
| [Twitter Create List](twitter/manage_lists.md#twitter-create-list) | This block creates a new Twitter List for the authenticated user |
| [Twitter Delete List](twitter/manage_lists.md#twitter-delete-list) | This block deletes a specified Twitter List owned by the authenticated user |
| [Twitter Delete Tweet](twitter/manage.md#twitter-delete-tweet) | This block deletes a tweet on Twitter |
| [Twitter Follow List](twitter/list_follows.md#twitter-follow-list) | This block follows a specified Twitter list for the authenticated user |
| [Twitter Follow User](twitter/follows.md#twitter-follow-user) | This block follows a specified Twitter user |
| [Twitter Get Blocked Users](twitter/blocks.md#twitter-get-blocked-users) | This block retrieves a list of users blocked by the authenticating user |
| [Twitter Get Bookmarked Tweets](twitter/bookmark.md#twitter-get-bookmarked-tweets) | This block retrieves bookmarked tweets from Twitter |
| [Twitter Get Followers](twitter/follows.md#twitter-get-followers) | This block retrieves followers of a specified Twitter user |
| [Twitter Get Following](twitter/follows.md#twitter-get-following) | This block retrieves the users that a specified Twitter user is following |
| [Twitter Get Home Timeline](twitter/timeline.md#twitter-get-home-timeline) | This block retrieves the authenticated user's home timeline |
| [Twitter Get Liked Tweets](twitter/like.md#twitter-get-liked-tweets) | This block gets information about tweets liked by a user |
| [Twitter Get Liking Users](twitter/like.md#twitter-get-liking-users) | This block gets information about users who liked a tweet |
| [Twitter Get List](twitter/list_lookup.md#twitter-get-list) | This block retrieves information about a specified Twitter List |
| [Twitter Get List Members](twitter/list_members.md#twitter-get-list-members) | This block retrieves the members of a specified Twitter List |
| [Twitter Get List Memberships](twitter/list_members.md#twitter-get-list-memberships) | This block retrieves all Lists that a specified user is a member of |
| [Twitter Get List Tweets](twitter/list_tweets_lookup.md#twitter-get-list-tweets) | This block retrieves tweets from a specified Twitter list |
| [Twitter Get Muted Users](twitter/mutes.md#twitter-get-muted-users) | This block gets a list of users muted by the authenticating user |
| [Twitter Get Owned Lists](twitter/list_lookup.md#twitter-get-owned-lists) | This block retrieves all Lists owned by a specified Twitter user |
| [Twitter Get Pinned Lists](twitter/pinned_lists.md#twitter-get-pinned-lists) | This block returns the Lists pinned by the authenticated user |
| [Twitter Get Quote Tweets](twitter/quote.md#twitter-get-quote-tweets) | This block gets quote tweets for a specific tweet |
| [Twitter Get Retweeters](twitter/retweet.md#twitter-get-retweeters) | This block gets information about who has retweeted a tweet |
| [Twitter Get Space Buyers](twitter/spaces_lookup.md#twitter-get-space-buyers) | This block retrieves a list of users who purchased tickets to a Twitter Space |
| [Twitter Get Space By Id](twitter/spaces_lookup.md#twitter-get-space-by-id) | This block retrieves information about a single Twitter Space |
| [Twitter Get Space Tweets](twitter/spaces_lookup.md#twitter-get-space-tweets) | This block retrieves tweets shared in a Twitter Space |
| [Twitter Get Spaces](twitter/spaces_lookup.md#twitter-get-spaces) | This block retrieves information about multiple Twitter Spaces |
| [Twitter Get Tweet](twitter/tweet_lookup.md#twitter-get-tweet) | This block retrieves information about a specific Tweet |
| [Twitter Get Tweets](twitter/tweet_lookup.md#twitter-get-tweets) | This block retrieves information about multiple Tweets |
| [Twitter Get User](twitter/user_lookup.md#twitter-get-user) | This block retrieves information about a specified Twitter user |
| [Twitter Get User Mentions](twitter/timeline.md#twitter-get-user-mentions) | This block retrieves Tweets mentioning a specific user |
| [Twitter Get User Tweets](twitter/timeline.md#twitter-get-user-tweets) | This block retrieves Tweets composed by a single user |
| [Twitter Get Users](twitter/user_lookup.md#twitter-get-users) | This block retrieves information about multiple Twitter users |
| [Twitter Hide Reply](twitter/hide.md#twitter-hide-reply) | This block hides a reply to a tweet |
| [Twitter Like Tweet](twitter/like.md#twitter-like-tweet) | This block likes a tweet |
| [Twitter Mute User](twitter/mutes.md#twitter-mute-user) | This block mutes a specified Twitter user |
| [Twitter Pin List](twitter/pinned_lists.md#twitter-pin-list) | This block allows the authenticated user to pin a specified List |
| [Twitter Post Tweet](twitter/manage.md#twitter-post-tweet) | This block posts a tweet on Twitter |
| [Twitter Remove Bookmark Tweet](twitter/bookmark.md#twitter-remove-bookmark-tweet) | This block removes a bookmark from a tweet on Twitter |
| [Twitter Remove List Member](twitter/list_members.md#twitter-remove-list-member) | This block removes a specified user from a Twitter List owned by the authenticated user |
| [Twitter Remove Retweet](twitter/retweet.md#twitter-remove-retweet) | This block removes a retweet on Twitter |
| [Twitter Retweet](twitter/retweet.md#twitter-retweet) | This block retweets a tweet on Twitter |
| [Twitter Search Recent Tweets](twitter/manage.md#twitter-search-recent-tweets) | This block searches all public Tweets in Twitter history |
| [Twitter Search Spaces](twitter/search_spaces.md#twitter-search-spaces) | This block searches for Twitter Spaces based on specified terms |
| [Twitter Unfollow List](twitter/list_follows.md#twitter-unfollow-list) | This block unfollows a specified Twitter list for the authenticated user |
| [Twitter Unfollow User](twitter/follows.md#twitter-unfollow-user) | This block unfollows a specified Twitter user |
| [Twitter Unhide Reply](twitter/hide.md#twitter-unhide-reply) | This block unhides a reply to a tweet |
| [Twitter Unlike Tweet](twitter/like.md#twitter-unlike-tweet) | This block unlikes a tweet |
| [Twitter Unmute User](twitter/mutes.md#twitter-unmute-user) | This block unmutes a specified Twitter user |
| [Twitter Unpin List](twitter/pinned_lists.md#twitter-unpin-list) | This block allows the authenticated user to unpin a specified List |
| [Twitter Update List](twitter/manage_lists.md#twitter-update-list) | This block updates a specified Twitter List owned by the authenticated user |

## Communication

| Block Name | Description |
|------------|-------------|
| [Baas Bot Join Meeting](baas/bots.md#baas-bot-join-meeting) | Deploy a bot to join and record a meeting |
| [Baas Bot Leave Meeting](baas/bots.md#baas-bot-leave-meeting) | Remove a bot from an ongoing meeting |
| [Gmail Add Label](google/gmail.md#gmail-add-label) | A block that adds a label to a specific email message in Gmail, creating the label if it doesn't exist |
| [Gmail Create Draft](google/gmail.md#gmail-create-draft) | Create draft emails in Gmail with automatic HTML detection and proper text formatting |
| [Gmail Draft Reply](google/gmail.md#gmail-draft-reply) | Create draft replies to Gmail threads with automatic HTML detection and proper text formatting |
| [Gmail Forward](google/gmail.md#gmail-forward) | Forward Gmail messages to other recipients with automatic HTML detection and proper formatting |
| [Gmail Get Profile](google/gmail.md#gmail-get-profile) | Get the authenticated user's Gmail profile details including email address and message statistics |
| [Gmail Get Thread](google/gmail.md#gmail-get-thread) | A block that retrieves an entire Gmail thread (email conversation) by ID, returning all messages with decoded bodies for reading complete conversations |
| [Gmail List Labels](google/gmail.md#gmail-list-labels) | A block that retrieves all labels (categories) from a Gmail account for organizing and categorizing emails |
| [Gmail Read](google/gmail.md#gmail-read) | A block that retrieves and reads emails from a Gmail account based on search criteria, returning detailed message information including subject, sender, body, and attachments |
| [Gmail Remove Label](google/gmail.md#gmail-remove-label) | A block that removes a label from a specific email message in a Gmail account |
| [Gmail Reply](google/gmail.md#gmail-reply) | Reply to Gmail threads with automatic HTML detection and proper text formatting |
| [Gmail Send](google/gmail.md#gmail-send) | Send emails via Gmail with automatic HTML detection and proper text formatting |
| [Hub Spot Engagement](hubspot/engagement.md#hub-spot-engagement) | Manages HubSpot engagements - sends emails and tracks engagement metrics |

## Developer Tools

| Block Name | Description |
|------------|-------------|
| [Exa Code Context](exa/code_context.md#exa-code-context) | Search billions of GitHub repos, docs, and Stack Overflow for relevant code examples |
| [Execute Code](misc.md#execute-code) | Executes code in a sandbox environment with internet access |
| [Execute Code Step](misc.md#execute-code-step) | Execute code in a previously instantiated sandbox |
| [Github Add Label](github/issues.md#github-add-label) | A block that adds a label to a GitHub issue or pull request for categorization and organization |
| [Github Assign Issue](github/issues.md#github-assign-issue) | A block that assigns a GitHub user to an issue for task ownership and tracking |
| [Github Assign PR Reviewer](github/pull_requests.md#github-assign-pr-reviewer) | This block assigns a reviewer to a specified GitHub pull request |
| [Github Comment](github/issues.md#github-comment) | A block that posts comments on GitHub issues or pull requests using the GitHub API |
| [Github Create Check Run](github/checks.md#github-create-check-run) | Creates a new check run for a specific commit in a GitHub repository |
| [Github Create Comment Object](github/reviews.md#github-create-comment-object) | Creates a comment object for use with GitHub blocks |
| [Github Create File](github/repo.md#github-create-file) | This block creates a new file in a GitHub repository |
| [Github Create PR Review](github/reviews.md#github-create-pr-review) | This block creates a review on a GitHub pull request with optional inline comments |
| [Github Create Repository](github/repo.md#github-create-repository) | This block creates a new GitHub repository |
| [Github Create Status](github/statuses.md#github-create-status) | Creates a new commit status in a GitHub repository |
| [Github Delete Branch](github/repo.md#github-delete-branch) | This block deletes a specified branch |
| [Github Discussion Trigger](github/triggers.md#github-discussion-trigger) | This block triggers on GitHub Discussions events |
| [Github Get CI Results](github/ci.md#github-get-ci-results) | This block gets CI results for a commit or PR, with optional search for specific errors/warnings in logs |
| [Github Get PR Review Comments](github/reviews.md#github-get-pr-review-comments) | This block gets all review comments from a GitHub pull request or from a specific review |
| [Github Issues Trigger](github/triggers.md#github-issues-trigger) | This block triggers on GitHub issues events |
| [Github List Branches](github/repo.md#github-list-branches) | This block lists all branches for a specified GitHub repository |
| [Github List Comments](github/issues.md#github-list-comments) | A block that retrieves all comments from a GitHub issue or pull request, including comment metadata and content |
| [Github List Discussions](github/repo.md#github-list-discussions) | This block lists recent discussions for a specified GitHub repository |
| [Github List Issues](github/issues.md#github-list-issues) | A block that retrieves a list of issues from a GitHub repository with their titles and URLs |
| [Github List PR Reviewers](github/pull_requests.md#github-list-pr-reviewers) | This block lists all reviewers for a specified GitHub pull request |
| [Github List PR Reviews](github/reviews.md#github-list-pr-reviews) | This block lists all reviews for a specified GitHub pull request |
| [Github List Pull Requests](github/pull_requests.md#github-list-pull-requests) | This block lists all pull requests for a specified GitHub repository |
| [Github List Releases](github/repo.md#github-list-releases) | This block lists all releases for a specified GitHub repository |
| [Github List Stargazers](github/repo.md#github-list-stargazers) | This block lists all users who have starred a specified GitHub repository |
| [Github List Tags](github/repo.md#github-list-tags) | This block lists all tags for a specified GitHub repository |
| [Github Make Branch](github/repo.md#github-make-branch) | This block creates a new branch from a specified source branch |
| [Github Make Issue](github/issues.md#github-make-issue) | A block that creates new issues on GitHub repositories with a title and body content |
| [Github Make Pull Request](github/pull_requests.md#github-make-pull-request) | This block creates a new pull request on a specified GitHub repository |
| [Github Pull Request Trigger](github/triggers.md#github-pull-request-trigger) | This block triggers on pull request events and outputs the event type and payload |
| [Github Read File](github/repo.md#github-read-file) | This block reads the content of a specified file from a GitHub repository |
| [Github Read Folder](github/repo.md#github-read-folder) | This block reads the content of a specified folder from a GitHub repository |
| [Github Read Issue](github/issues.md#github-read-issue) | A block that retrieves information about a specific GitHub issue, including its title, body content, and creator |
| [Github Read Pull Request](github/pull_requests.md#github-read-pull-request) | This block reads the body, title, user, and changes of a specified GitHub pull request |
| [Github Release Trigger](github/triggers.md#github-release-trigger) | This block triggers on GitHub release events |
| [Github Remove Label](github/issues.md#github-remove-label) | A block that removes a label from a GitHub issue or pull request |
| [Github Resolve Review Discussion](github/reviews.md#github-resolve-review-discussion) | This block resolves or unresolves a review discussion thread on a GitHub pull request |
| [Github Star Trigger](github/triggers.md#github-star-trigger) | This block triggers on GitHub star events |
| [Github Submit Pending Review](github/reviews.md#github-submit-pending-review) | This block submits a pending (draft) review on a GitHub pull request |
| [Github Unassign Issue](github/issues.md#github-unassign-issue) | A block that removes a user's assignment from a GitHub issue |
| [Github Unassign PR Reviewer](github/pull_requests.md#github-unassign-pr-reviewer) | This block unassigns a reviewer from a specified GitHub pull request |
| [Github Update Check Run](github/checks.md#github-update-check-run) | Updates an existing check run in a GitHub repository |
| [Github Update Comment](github/issues.md#github-update-comment) | A block that updates an existing comment on a GitHub issue or pull request |
| [Github Update File](github/repo.md#github-update-file) | This block updates an existing file in a GitHub repository |
| [Instantiate Code Sandbox](misc.md#instantiate-code-sandbox) | Instantiate a sandbox environment with internet access in which you can execute code with the Execute Code Step block |
| [Slant3D Order Webhook](slant3d/webhook.md#slant3d-order-webhook) | This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped |

## Media Generation

| Block Name | Description |
|------------|-------------|
| [Add Audio To Video](multimedia.md#add-audio-to-video) | Block to attach an audio file to a video file using moviepy |
| [Loop Video](multimedia.md#loop-video) | Block to loop a video to a given duration or number of repeats |
| [Media Duration](multimedia.md#media-duration) | Block to get the duration of a media file |

## Productivity

| Block Name | Description |
|------------|-------------|
| [Google Calendar Create Event](google/calendar.md#google-calendar-create-event) | This block creates a new event in Google Calendar with customizable parameters |
| [Notion Create Page](notion/create_page.md#notion-create-page) | Create a new page in Notion |
| [Notion Read Database](notion/read_database.md#notion-read-database) | Query a Notion database with optional filtering and sorting, returning structured entries |
| [Notion Read Page](notion/read_page.md#notion-read-page) | Read a Notion page by its ID and return its raw JSON |
| [Notion Read Page Markdown](notion/read_page_markdown.md#notion-read-page-markdown) | Read a Notion page and convert it to Markdown format with proper formatting for headings, lists, links, and rich text |
| [Notion Search](notion/search.md#notion-search) | Search your Notion workspace for pages and databases by text query |
| [Todoist Close Task](todoist/tasks.md#todoist-close-task) | Closes a task in Todoist |
| [Todoist Create Comment](todoist/comments.md#todoist-create-comment) | Creates a new comment on a Todoist task or project |
| [Todoist Create Label](todoist/labels.md#todoist-create-label) | Creates a new label in Todoist, It will not work if same name already exists |
| [Todoist Create Project](todoist/projects.md#todoist-create-project) | Creates a new project in Todoist |
| [Todoist Create Task](todoist/tasks.md#todoist-create-task) | Creates a new task in a Todoist project |
| [Todoist Delete Comment](todoist/comments.md#todoist-delete-comment) | Deletes a Todoist comment |
| [Todoist Delete Label](todoist/labels.md#todoist-delete-label) | Deletes a personal label in Todoist |
| [Todoist Delete Project](todoist/projects.md#todoist-delete-project) | Deletes a Todoist project and all its contents |
| [Todoist Delete Section](todoist/sections.md#todoist-delete-section) | Deletes a section and all its tasks from Todoist |
| [Todoist Delete Task](todoist/tasks.md#todoist-delete-task) | Deletes a task in Todoist |
| [Todoist Get Comment](todoist/comments.md#todoist-get-comment) | Get a single comment from Todoist |
| [Todoist Get Comments](todoist/comments.md#todoist-get-comments) | Get all comments for a Todoist task or project |
| [Todoist Get Label](todoist/labels.md#todoist-get-label) | Gets a personal label from Todoist by ID |
| [Todoist Get Project](todoist/projects.md#todoist-get-project) | Gets details for a specific Todoist project |
| [Todoist Get Section](todoist/sections.md#todoist-get-section) | Gets a single section by ID from Todoist |
| [Todoist Get Shared Labels](todoist/labels.md#todoist-get-shared-labels) | Gets all shared labels from Todoist |
| [Todoist Get Task](todoist/tasks.md#todoist-get-task) | Get an active task from Todoist |
| [Todoist Get Tasks](todoist/tasks.md#todoist-get-tasks) | Get active tasks from Todoist |
| [Todoist List Collaborators](todoist/projects.md#todoist-list-collaborators) | Gets all collaborators for a specific Todoist project |
| [Todoist List Labels](todoist/labels.md#todoist-list-labels) | Gets all personal labels from Todoist |
| [Todoist List Projects](todoist/projects.md#todoist-list-projects) | Gets all projects and their details from Todoist |
| [Todoist List Sections](todoist/sections.md#todoist-list-sections) | Gets all sections and their details from Todoist |
| [Todoist Remove Shared Labels](todoist/labels.md#todoist-remove-shared-labels) | Removes all instances of a shared label |
| [Todoist Rename Shared Labels](todoist/labels.md#todoist-rename-shared-labels) | Renames all instances of a shared label |
| [Todoist Reopen Task](todoist/tasks.md#todoist-reopen-task) | Reopens a task in Todoist |
| [Todoist Update Comment](todoist/comments.md#todoist-update-comment) | Updates a Todoist comment |
| [Todoist Update Label](todoist/labels.md#todoist-update-label) | Updates a personal label in Todoist |
| [Todoist Update Project](todoist/projects.md#todoist-update-project) | Updates an existing project in Todoist |
| [Todoist Update Task](todoist/tasks.md#todoist-update-task) | Updates an existing task in Todoist |

## Logic and Control Flow

| Block Name | Description |
|------------|-------------|
| [Calculator](logic.md#calculator) | Performs a mathematical operation on two numbers |
| [Condition](logic.md#condition) | Handles conditional logic based on comparison operators |
| [Count Items](logic.md#count-items) | Counts the number of items in a collection |
| [Data Sampling](logic.md#data-sampling) | This block samples data from a given dataset using various sampling methods |
| [Exa Webset Ready Check](exa/websets.md#exa-webset-ready-check) | Check if webset is ready for next operation - enables conditional workflow branching |
| [If Input Matches](logic.md#if-input-matches) | Handles conditional logic based on comparison operators |
| [Pinecone Init](logic.md#pinecone-init) | Initializes a Pinecone index |
| [Pinecone Insert](logic.md#pinecone-insert) | Upload data to a Pinecone index |
| [Pinecone Query](logic.md#pinecone-query) | Queries a Pinecone index |
| [Step Through Items](logic.md#step-through-items) | Iterates over a list or dictionary and outputs each item |

## Input/Output

| Block Name | Description |
|------------|-------------|
| [Exa Webset Webhook](exa/webhook_blocks.md#exa-webset-webhook) | Receive webhook notifications for Exa webset events |
| [Generic Webhook Trigger](generic_webhook/triggers.md#generic-webhook-trigger) | This block will output the contents of the generic input for the webhook |
| [Read RSS Feed](misc.md#read-rss-feed) | Reads RSS feed entries from a given URL |
| [Send Authenticated Web Request](misc.md#send-authenticated-web-request) | Make an authenticated HTTP request with host-scoped credentials (JSON / form / multipart) |
| [Send Email](misc.md#send-email) | This block sends an email using the provided SMTP credentials |
| [Send Web Request](misc.md#send-web-request) | Make an HTTP request (JSON / form / multipart) |

## Agent Integration

| Block Name | Description |
|------------|-------------|
| [Agent Executor](misc.md#agent-executor) | Executes an existing agent inside your agent |

## CRM Services

| Block Name | Description |
|------------|-------------|
| [Add Lead To Campaign](smartlead/campaign.md#add-lead-to-campaign) | Add a lead to a campaign in SmartLead |
| [Create Campaign](smartlead/campaign.md#create-campaign) | Create a campaign in SmartLead |
| [Hub Spot Company](hubspot/company.md#hub-spot-company) | Manages HubSpot companies - create, update, and retrieve company information |
| [Hub Spot Contact](hubspot/contact.md#hub-spot-contact) | Manages HubSpot contacts - create, update, and retrieve contact information |
| [Save Campaign Sequences](smartlead/campaign.md#save-campaign-sequences) | Save sequences within a campaign |

## AI Safety

| Block Name | Description |
|------------|-------------|
| [Nvidia Deepfake Detect](nvidia/deepfake.md#nvidia-deepfake-detect) | Detects potential deepfakes in images using Nvidia's AI API |

## Issue Tracking

| Block Name | Description |
|------------|-------------|
| [Linear Create Comment](linear/comment.md#linear-create-comment) | Creates a new comment on a Linear issue |
| [Linear Create Issue](linear/issues.md#linear-create-issue) | Creates a new issue on Linear |
| [Linear Get Project Issues](linear/issues.md#linear-get-project-issues) | Gets issues from a Linear project filtered by status and assignee |
| [Linear Search Projects](linear/projects.md#linear-search-projects) | Searches for projects on Linear |

## Hardware

| Block Name | Description |
|------------|-------------|
| [Compass AI Trigger](compass/triggers.md#compass-ai-trigger) | This block will output the contents of the compass transcription |
