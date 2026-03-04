---
layout:
  width: default
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: false
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: true
---

# AutoGPT Blocks Overview

AutoGPT uses a modular approach with various "blocks" to handle different tasks. These blocks are the building blocks of AutoGPT workflows, allowing users to create complex automations by combining simple, specialized components.

{% hint style="info" %}
**Creating Your Own Blocks**

Want to create your own custom blocks? Check out our guides:

* [Build your own Blocks](https://docs.agpt.co/platform/new_blocks/) - Step-by-step tutorial with examples
* [Block SDK Guide](https://docs.agpt.co/platform/block-sdk-guide/) - Advanced SDK patterns with OAuth, webhooks, and provider configuration
{% endhint %}

Below is a comprehensive list of all available blocks, categorized by their primary function. Click on any block name to view its detailed documentation.

## Basic Operations

| Block Name | Description |
|------------|-------------|
| [Add Memory](block-integrations/basic.md#add-memory) | Add new memories to Mem0 with user segmentation |
| [Add To Dictionary](block-integrations/basic.md#add-to-dictionary) | Adds a new key-value pair to a dictionary |
| [Add To Library From Store](block-integrations/system/library_operations.md#add-to-library-from-store) | Add an agent from the store to your personal library |
| [Add To List](block-integrations/basic.md#add-to-list) | Adds a new entry to a list |
| [Agent Date Input](block-integrations/basic.md#agent-date-input) | Block for date input |
| [Agent Dropdown Input](block-integrations/basic.md#agent-dropdown-input) | Block for dropdown text selection |
| [Agent File Input](block-integrations/basic.md#agent-file-input) | Block for file upload input (string path for example) |
| [Agent Google Drive File Input](block-integrations/basic.md#agent-google-drive-file-input) | Block for selecting a file from Google Drive |
| [Agent Input](block-integrations/basic.md#agent-input) | A block that accepts and processes user input values within a workflow, supporting various input types and validation |
| [Agent Long Text Input](block-integrations/basic.md#agent-long-text-input) | Block for long text input (multi-line) |
| [Agent Number Input](block-integrations/basic.md#agent-number-input) | Block for number input |
| [Agent Output](block-integrations/basic.md#agent-output) | A block that records and formats workflow results for display to users, with optional Jinja2 template formatting support |
| [Agent Short Text Input](block-integrations/basic.md#agent-short-text-input) | Block for short text input (single-line) |
| [Agent Table Input](block-integrations/basic.md#agent-table-input) | Block for table data input with customizable headers |
| [Agent Time Input](block-integrations/basic.md#agent-time-input) | Block for time input |
| [Agent Toggle Input](block-integrations/basic.md#agent-toggle-input) | Block for boolean toggle input |
| [Block Installation](block-integrations/basic.md#block-installation) | Given a code string, this block allows the verification and installation of a block code into the system |
| [Concatenate Lists](block-integrations/basic.md#concatenate-lists) | Concatenates multiple lists into a single list |
| [Dictionary Is Empty](block-integrations/basic.md#dictionary-is-empty) | Checks if a dictionary is empty |
| [File Store](block-integrations/basic.md#file-store) | Downloads and stores a file from a URL, data URI, or local path |
| [Find In Dictionary](block-integrations/basic.md#find-in-dictionary) | A block that looks up a value in a dictionary, list, or object by key or index and returns the corresponding value |
| [Find In List](block-integrations/basic.md#find-in-list) | Finds the index of the value in the list |
| [Flatten List](block-integrations/basic.md#flatten-list) | Flattens a nested list structure into a single flat list |
| [Get All Memories](block-integrations/basic.md#get-all-memories) | Retrieve all memories from Mem0 with optional conversation filtering |
| [Get Latest Memory](block-integrations/basic.md#get-latest-memory) | Retrieve the latest memory from Mem0 with optional key filtering |
| [Get List Item](block-integrations/basic.md#get-list-item) | Returns the element at the given index |
| [Get Store Agent Details](block-integrations/system/store_operations.md#get-store-agent-details) | Get detailed information about an agent from the store |
| [Get Weather Information](block-integrations/basic.md#get-weather-information) | Retrieves weather information for a specified location using OpenWeatherMap API |
| [Human In The Loop](block-integrations/basic.md#human-in-the-loop) | Pause execution for human review |
| [Interleave Lists](block-integrations/basic.md#interleave-lists) | Interleaves elements from multiple lists in round-robin fashion, alternating between sources |
| [List Difference](block-integrations/basic.md#list-difference) | Computes the difference between two lists |
| [List Intersection](block-integrations/basic.md#list-intersection) | Computes the intersection of two lists, returning only elements present in both |
| [List Is Empty](block-integrations/basic.md#list-is-empty) | Checks if a list is empty |
| [List Library Agents](block-integrations/system/library_operations.md#list-library-agents) | List all agents in your personal library |
| [Note](block-integrations/basic.md#note) | A visual annotation block that displays a sticky note in the workflow editor for documentation and organization purposes |
| [Print To Console](block-integrations/basic.md#print-to-console) | A debugging block that outputs text to the console for monitoring and troubleshooting workflow execution |
| [Remove From Dictionary](block-integrations/basic.md#remove-from-dictionary) | Removes a key-value pair from a dictionary |
| [Remove From List](block-integrations/basic.md#remove-from-list) | Removes an item from a list by value or index |
| [Replace Dictionary Value](block-integrations/basic.md#replace-dictionary-value) | Replaces the value for a specified key in a dictionary |
| [Replace List Item](block-integrations/basic.md#replace-list-item) | Replaces an item at the specified index |
| [Reverse List Order](block-integrations/basic.md#reverse-list-order) | Reverses the order of elements in a list |
| [Search Memory](block-integrations/basic.md#search-memory) | Search memories in Mem0 by user |
| [Search Store Agents](block-integrations/system/store_operations.md#search-store-agents) | Search for agents in the store |
| [Slant3D Cancel Order](block-integrations/slant3d/order.md#slant3d-cancel-order) | Cancel an existing order |
| [Slant3D Create Order](block-integrations/slant3d/order.md#slant3d-create-order) | Create a new print order |
| [Slant3D Estimate Order](block-integrations/slant3d/order.md#slant3d-estimate-order) | Get order cost estimate |
| [Slant3D Estimate Shipping](block-integrations/slant3d/order.md#slant3d-estimate-shipping) | Get shipping cost estimate |
| [Slant3D Filament](block-integrations/slant3d/filament.md#slant3d-filament) | Get list of available filaments |
| [Slant3D Get Orders](block-integrations/slant3d/order.md#slant3d-get-orders) | Get all orders for the account |
| [Slant3D Slicer](block-integrations/slant3d/slicing.md#slant3d-slicer) | Slice a 3D model file and get pricing information |
| [Slant3D Tracking](block-integrations/slant3d/order.md#slant3d-tracking) | Track order status and shipping |
| [Store Value](block-integrations/basic.md#store-value) | A basic block that stores and forwards a value throughout workflows, allowing it to be reused without changes across multiple blocks |
| [Universal Type Converter](block-integrations/basic.md#universal-type-converter) | This block is used to convert a value to a universal type |
| [XML Parser](block-integrations/basic.md#xml-parser) | Parses XML using gravitasml to tokenize and coverts it to dict |
| [Zip Lists](block-integrations/basic.md#zip-lists) | Zips multiple lists together into a list of grouped elements |

## Data Processing

| Block Name | Description |
|------------|-------------|
| [Airtable Create Base](block-integrations/airtable/bases.md#airtable-create-base) | Create or find a base in Airtable |
| [Airtable Create Field](block-integrations/airtable/schema.md#airtable-create-field) | Add a new field to an Airtable table |
| [Airtable Create Records](block-integrations/airtable/records.md#airtable-create-records) | Create records in an Airtable table |
| [Airtable Create Table](block-integrations/airtable/schema.md#airtable-create-table) | Create a new table in an Airtable base |
| [Airtable Delete Records](block-integrations/airtable/records.md#airtable-delete-records) | Delete records from an Airtable table |
| [Airtable Get Record](block-integrations/airtable/records.md#airtable-get-record) | Get a single record from Airtable |
| [Airtable List Bases](block-integrations/airtable/bases.md#airtable-list-bases) | List all bases in Airtable |
| [Airtable List Records](block-integrations/airtable/records.md#airtable-list-records) | List records from an Airtable table |
| [Airtable List Schema](block-integrations/airtable/schema.md#airtable-list-schema) | Get the complete schema of an Airtable base |
| [Airtable Update Field](block-integrations/airtable/schema.md#airtable-update-field) | Update field properties in an Airtable table |
| [Airtable Update Records](block-integrations/airtable/records.md#airtable-update-records) | Update records in an Airtable table |
| [Airtable Update Table](block-integrations/airtable/schema.md#airtable-update-table) | Update table properties |
| [Airtable Webhook Trigger](block-integrations/airtable/triggers.md#airtable-webhook-trigger) | Starts a flow whenever Airtable emits a webhook event |
| [Baas Bot Delete Recording](block-integrations/baas/bots.md#baas-bot-delete-recording) | Permanently delete a meeting's recorded data |
| [Baas Bot Fetch Meeting Data](block-integrations/baas/bots.md#baas-bot-fetch-meeting-data) | Retrieve recorded meeting data |
| [Create Dictionary](block-integrations/data.md#create-dictionary) | Creates a dictionary with the specified key-value pairs |
| [Create List](block-integrations/data.md#create-list) | Creates a list with the specified values |
| [Data For Seo Keyword Suggestions](block-integrations/dataforseo/keyword_suggestions.md#data-for-seo-keyword-suggestions) | Get keyword suggestions from DataForSEO Labs Google API |
| [Data For Seo Related Keywords](block-integrations/dataforseo/related_keywords.md#data-for-seo-related-keywords) | Get related keywords from DataForSEO Labs Google API |
| [Exa Create Import](block-integrations/exa/websets_import_export.md#exa-create-import) | Import CSV data to use with websets for targeted searches |
| [Exa Delete Import](block-integrations/exa/websets_import_export.md#exa-delete-import) | Delete an import |
| [Exa Export Webset](block-integrations/exa/websets_import_export.md#exa-export-webset) | Export webset data in JSON, CSV, or JSON Lines format |
| [Exa Get Import](block-integrations/exa/websets_import_export.md#exa-get-import) | Get the status and details of an import |
| [Exa Get New Items](block-integrations/exa/websets_items.md#exa-get-new-items) | Get items added since a cursor - enables incremental processing without reprocessing |
| [Exa List Imports](block-integrations/exa/websets_import_export.md#exa-list-imports) | List all imports with pagination support |
| [File Read](block-integrations/data.md#file-read) | Reads a file and returns its content as a string, with optional chunking by delimiter and size limits |
| [Google Calendar Read Events](block-integrations/google/calendar.md#google-calendar-read-events) | Retrieves upcoming events from a Google Calendar with filtering options |
| [Google Docs Append Markdown](block-integrations/google/docs.md#google-docs-append-markdown) | Append Markdown content to the end of a Google Doc with full formatting - ideal for LLM/AI output |
| [Google Docs Append Plain Text](block-integrations/google/docs.md#google-docs-append-plain-text) | Append plain text to the end of a Google Doc (no formatting applied) |
| [Google Docs Create](block-integrations/google/docs.md#google-docs-create) | Create a new Google Doc |
| [Google Docs Delete Content](block-integrations/google/docs.md#google-docs-delete-content) | Delete a range of content from a Google Doc |
| [Google Docs Export](block-integrations/google/docs.md#google-docs-export) | Export a Google Doc to PDF, Word, text, or other formats |
| [Google Docs Find Replace Plain Text](block-integrations/google/docs.md#google-docs-find-replace-plain-text) | Find and replace plain text in a Google Doc (no formatting applied to replacement) |
| [Google Docs Format Text](block-integrations/google/docs.md#google-docs-format-text) | Apply formatting (bold, italic, color, etc |
| [Google Docs Get Metadata](block-integrations/google/docs.md#google-docs-get-metadata) | Get metadata about a Google Doc |
| [Google Docs Get Structure](block-integrations/google/docs.md#google-docs-get-structure) | Get document structure with index positions for precise editing operations |
| [Google Docs Insert Markdown At](block-integrations/google/docs.md#google-docs-insert-markdown-at) | Insert formatted Markdown at a specific position in a Google Doc - ideal for LLM/AI output |
| [Google Docs Insert Page Break](block-integrations/google/docs.md#google-docs-insert-page-break) | Insert a page break into a Google Doc |
| [Google Docs Insert Plain Text](block-integrations/google/docs.md#google-docs-insert-plain-text) | Insert plain text at a specific position in a Google Doc (no formatting applied) |
| [Google Docs Insert Table](block-integrations/google/docs.md#google-docs-insert-table) | Insert a table into a Google Doc, optionally with content and Markdown formatting |
| [Google Docs Read](block-integrations/google/docs.md#google-docs-read) | Read text content from a Google Doc |
| [Google Docs Replace All With Markdown](block-integrations/google/docs.md#google-docs-replace-all-with-markdown) | Replace entire Google Doc content with formatted Markdown - ideal for LLM/AI output |
| [Google Docs Replace Content With Markdown](block-integrations/google/docs.md#google-docs-replace-content-with-markdown) | Find text and replace it with formatted Markdown - ideal for LLM/AI output and templates |
| [Google Docs Replace Range With Markdown](block-integrations/google/docs.md#google-docs-replace-range-with-markdown) | Replace a specific index range in a Google Doc with formatted Markdown - ideal for LLM/AI output |
| [Google Docs Set Public Access](block-integrations/google/docs.md#google-docs-set-public-access) | Make a Google Doc public or private |
| [Google Docs Share](block-integrations/google/docs.md#google-docs-share) | Share a Google Doc with specific users |
| [Google Sheets Add Column](block-integrations/google/sheets.md#google-sheets-add-column) | Add a new column with a header |
| [Google Sheets Add Dropdown](block-integrations/google/sheets.md#google-sheets-add-dropdown) | Add a dropdown list (data validation) to cells |
| [Google Sheets Add Note](block-integrations/google/sheets.md#google-sheets-add-note) | Add a note to a cell in a Google Sheet |
| [Google Sheets Append Row](block-integrations/google/sheets.md#google-sheets-append-row) | Append or Add a single row to the end of a Google Sheet |
| [Google Sheets Batch Operations](block-integrations/google/sheets.md#google-sheets-batch-operations) | This block performs multiple operations on a Google Sheets spreadsheet in a single batch request |
| [Google Sheets Clear](block-integrations/google/sheets.md#google-sheets-clear) | This block clears data from a specified range in a Google Sheets spreadsheet |
| [Google Sheets Copy To Spreadsheet](block-integrations/google/sheets.md#google-sheets-copy-to-spreadsheet) | Copy a sheet from one spreadsheet to another |
| [Google Sheets Create Named Range](block-integrations/google/sheets.md#google-sheets-create-named-range) | Create a named range to reference cells by name instead of A1 notation |
| [Google Sheets Create Spreadsheet](block-integrations/google/sheets.md#google-sheets-create-spreadsheet) | This block creates a new Google Sheets spreadsheet with specified sheets |
| [Google Sheets Delete Column](block-integrations/google/sheets.md#google-sheets-delete-column) | Delete a column by header name or column letter |
| [Google Sheets Delete Rows](block-integrations/google/sheets.md#google-sheets-delete-rows) | Delete specific rows from a Google Sheet by their row indices |
| [Google Sheets Export Csv](block-integrations/google/sheets.md#google-sheets-export-csv) | Export a Google Sheet as CSV data |
| [Google Sheets Filter Rows](block-integrations/google/sheets.md#google-sheets-filter-rows) | Filter rows in a Google Sheet based on a column condition |
| [Google Sheets Find](block-integrations/google/sheets.md#google-sheets-find) | Find text in a Google Sheets spreadsheet |
| [Google Sheets Find Replace](block-integrations/google/sheets.md#google-sheets-find-replace) | This block finds and replaces text in a Google Sheets spreadsheet |
| [Google Sheets Format](block-integrations/google/sheets.md#google-sheets-format) | Format a range in a Google Sheet (sheet optional) |
| [Google Sheets Get Column](block-integrations/google/sheets.md#google-sheets-get-column) | Extract all values from a specific column |
| [Google Sheets Get Notes](block-integrations/google/sheets.md#google-sheets-get-notes) | Get notes from cells in a Google Sheet |
| [Google Sheets Get Row](block-integrations/google/sheets.md#google-sheets-get-row) | Get a specific row by its index |
| [Google Sheets Get Row Count](block-integrations/google/sheets.md#google-sheets-get-row-count) | Get row count and dimensions of a Google Sheet |
| [Google Sheets Get Unique Values](block-integrations/google/sheets.md#google-sheets-get-unique-values) | Get unique values from a column |
| [Google Sheets Import Csv](block-integrations/google/sheets.md#google-sheets-import-csv) | Import CSV data into a Google Sheet |
| [Google Sheets Insert Row](block-integrations/google/sheets.md#google-sheets-insert-row) | Insert a single row at a specific position |
| [Google Sheets List Named Ranges](block-integrations/google/sheets.md#google-sheets-list-named-ranges) | List all named ranges in a spreadsheet |
| [Google Sheets Lookup Row](block-integrations/google/sheets.md#google-sheets-lookup-row) | Look up a row by finding a value in a specific column |
| [Google Sheets Manage Sheet](block-integrations/google/sheets.md#google-sheets-manage-sheet) | Create, delete, or copy sheets (sheet optional) |
| [Google Sheets Metadata](block-integrations/google/sheets.md#google-sheets-metadata) | This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties |
| [Google Sheets Protect Range](block-integrations/google/sheets.md#google-sheets-protect-range) | Protect a cell range or entire sheet from editing |
| [Google Sheets Read](block-integrations/google/sheets.md#google-sheets-read) | A block that reads data from a Google Sheets spreadsheet using A1 notation range selection |
| [Google Sheets Remove Duplicates](block-integrations/google/sheets.md#google-sheets-remove-duplicates) | Remove duplicate rows based on specified columns |
| [Google Sheets Set Public Access](block-integrations/google/sheets.md#google-sheets-set-public-access) | Make a Google Spreadsheet public or private |
| [Google Sheets Share Spreadsheet](block-integrations/google/sheets.md#google-sheets-share-spreadsheet) | Share a Google Spreadsheet with users or get shareable link |
| [Google Sheets Sort](block-integrations/google/sheets.md#google-sheets-sort) | Sort a Google Sheet by one or two columns |
| [Google Sheets Update Cell](block-integrations/google/sheets.md#google-sheets-update-cell) | Update a single cell in a Google Sheets spreadsheet |
| [Google Sheets Update Row](block-integrations/google/sheets.md#google-sheets-update-row) | Update a specific row by its index |
| [Google Sheets Write](block-integrations/google/sheets.md#google-sheets-write) | A block that writes data to a Google Sheets spreadsheet at a specified A1 notation range |
| [Keyword Suggestion Extractor](block-integrations/dataforseo/keyword_suggestions.md#keyword-suggestion-extractor) | Extract individual fields from a KeywordSuggestion object |
| [Persist Information](block-integrations/data.md#persist-information) | Persist key-value information for the current user |
| [Read Spreadsheet](block-integrations/data.md#read-spreadsheet) | Reads CSV and Excel files and outputs the data as a list of dictionaries and individual rows |
| [Related Keyword Extractor](block-integrations/dataforseo/related_keywords.md#related-keyword-extractor) | Extract individual fields from a RelatedKeyword object |
| [Retrieve Information](block-integrations/data.md#retrieve-information) | Retrieve key-value information for the current user |
| [Screenshot Web Page](block-integrations/data.md#screenshot-web-page) | Takes a screenshot of a specified website using ScreenshotOne API |

## Text Processing

| Block Name | Description |
|------------|-------------|
| [Code Extraction](block-integrations/text.md#code-extraction) | Extracts code blocks from text and identifies their programming languages |
| [Combine Texts](block-integrations/text.md#combine-texts) | This block combines multiple input texts into a single output text |
| [Countdown Timer](block-integrations/text.md#countdown-timer) | This block triggers after a specified duration |
| [Extract Text Information](block-integrations/text.md#extract-text-information) | This block extracts the text from the given text using the pattern (regex) |
| [Fill Text Template](block-integrations/text.md#fill-text-template) | This block formats the given texts using the format template |
| [Get Current Date](block-integrations/text.md#get-current-date) | This block outputs the current date with an optional offset |
| [Get Current Date And Time](block-integrations/text.md#get-current-date-and-time) | This block outputs the current date and time |
| [Get Current Time](block-integrations/text.md#get-current-time) | This block outputs the current time |
| [Match Text Pattern](block-integrations/text.md#match-text-pattern) | Matches text against a regex pattern and forwards data to positive or negative output based on the match |
| [Text Decoder](block-integrations/text.md#text-decoder) | Decodes a string containing escape sequences into actual text |
| [Text Encoder](block-integrations/text.md#text-encoder) | Encodes a string by converting special characters into escape sequences |
| [Text Replace](block-integrations/text.md#text-replace) | This block is used to replace a text with a new text |
| [Text Split](block-integrations/text.md#text-split) | This block is used to split a text into a list of strings |
| [Word Character Count](block-integrations/text.md#word-character-count) | Counts the number of words and characters in a given text |

## AI and Language Models

| Block Name | Description |
|------------|-------------|
| [AI Ad Maker Video Creator](block-integrations/llm.md#ai-ad-maker-video-creator) | Creates an AI‑generated 30‑second advert (text + images) |
| [AI Condition](block-integrations/llm.md#ai-condition) | Uses AI to evaluate natural language conditions and provide conditional outputs |
| [AI Conversation](block-integrations/llm.md#ai-conversation) | A block that facilitates multi-turn conversations with a Large Language Model (LLM), maintaining context across message exchanges |
| [AI Image Customizer](block-integrations/llm.md#ai-image-customizer) | Generate and edit custom images using Google's Nano-Banana model from Gemini 2 |
| [AI Image Editor](block-integrations/llm.md#ai-image-editor) | Edit images using BlackForest Labs' Flux Kontext models |
| [AI Image Generator](block-integrations/llm.md#ai-image-generator) | Generate images using various AI models through a unified interface |
| [AI List Generator](block-integrations/llm.md#ai-list-generator) | A block that creates lists of items based on prompts using a Large Language Model (LLM), with optional source data for context |
| [AI Music Generator](block-integrations/llm.md#ai-music-generator) | This block generates music using Meta's MusicGen model on Replicate |
| [AI Screenshot To Video Ad](block-integrations/llm.md#ai-screenshot-to-video-ad) | Turns a screenshot into an engaging, avatar‑narrated video advert |
| [AI Shortform Video Creator](block-integrations/llm.md#ai-shortform-video-creator) | Creates a shortform video using revid |
| [AI Structured Response Generator](block-integrations/llm.md#ai-structured-response-generator) | A block that generates structured JSON responses using a Large Language Model (LLM), with schema validation and format enforcement |
| [AI Text Generator](block-integrations/llm.md#ai-text-generator) | A block that produces text responses using a Large Language Model (LLM) based on customizable prompts and system instructions |
| [AI Text Summarizer](block-integrations/llm.md#ai-text-summarizer) | A block that summarizes long texts using a Large Language Model (LLM), with configurable focus topics and summary styles |
| [AI Video Generator](block-integrations/fal/ai_video_generator.md#ai-video-generator) | Generate videos using FAL AI models |
| [Bannerbear Text Overlay](block-integrations/bannerbear/text_overlay.md#bannerbear-text-overlay) | Add text overlay to images using Bannerbear templates |
| [Claude Code](block-integrations/llm.md#claude-code) | Execute tasks using Claude Code in an E2B sandbox |
| [Code Generation](block-integrations/llm.md#code-generation) | Generate or refactor code using OpenAI's Codex (Responses API) |
| [Create Talking Avatar Video](block-integrations/llm.md#create-talking-avatar-video) | This block integrates with D-ID to create video clips and retrieve their URLs |
| [Exa Answer](block-integrations/exa/answers.md#exa-answer) | Get an LLM answer to a question informed by Exa search results |
| [Exa Create Enrichment](block-integrations/exa/websets_enrichment.md#exa-create-enrichment) | Create enrichments to extract additional structured data from webset items |
| [Exa Create Research](block-integrations/exa/research.md#exa-create-research) | Create research task with optional waiting - explores web and synthesizes findings with citations |
| [Ideogram Model](block-integrations/llm.md#ideogram-model) | This block runs Ideogram models with both simple and advanced settings |
| [Jina Chunking](block-integrations/jina/chunking.md#jina-chunking) | Chunks texts using Jina AI's segmentation service |
| [Jina Embedding](block-integrations/jina/embeddings.md#jina-embedding) | Generates embeddings using Jina AI |
| [Perplexity](block-integrations/llm.md#perplexity) | Query Perplexity's sonar models with real-time web search capabilities and receive annotated responses with source citations |
| [Replicate Flux Advanced Model](block-integrations/replicate/flux_advanced.md#replicate-flux-advanced-model) | This block runs Flux models on Replicate with advanced settings |
| [Replicate Model](block-integrations/replicate/replicate_block.md#replicate-model) | Run Replicate models synchronously |
| [Smart Decision Maker](block-integrations/llm.md#smart-decision-maker) | Uses AI to intelligently decide what tool to use |
| [Stagehand Act](block-integrations/stagehand/blocks.md#stagehand-act) | Interact with a web page by performing actions on a web page |
| [Stagehand Extract](block-integrations/stagehand/blocks.md#stagehand-extract) | Extract structured data from a webpage |
| [Stagehand Observe](block-integrations/stagehand/blocks.md#stagehand-observe) | Find suggested actions for your workflows |
| [Unreal Text To Speech](block-integrations/llm.md#unreal-text-to-speech) | Converts text to speech using the Unreal Speech API |
| [Video Narration](block-integrations/video/narration.md#video-narration) | Generate AI narration and add to video |

## Search and Information Retrieval

| Block Name | Description |
|------------|-------------|
| [Ask Wolfram](block-integrations/wolfram/llm_api.md#ask-wolfram) | Ask Wolfram Alpha a question |
| [Exa Bulk Webset Items](block-integrations/exa/websets_items.md#exa-bulk-webset-items) | Get all items from a webset in bulk (with configurable limits) |
| [Exa Cancel Enrichment](block-integrations/exa/websets_enrichment.md#exa-cancel-enrichment) | Cancel a running enrichment operation |
| [Exa Cancel Webset](block-integrations/exa/websets.md#exa-cancel-webset) | Cancel all operations being performed on a Webset |
| [Exa Cancel Webset Search](block-integrations/exa/websets_search.md#exa-cancel-webset-search) | Cancel a running webset search |
| [Exa Contents](block-integrations/exa/contents.md#exa-contents) | Retrieves document contents using Exa's contents API |
| [Exa Create Monitor](block-integrations/exa/websets_monitor.md#exa-create-monitor) | Create automated monitors to keep websets updated with fresh data on a schedule |
| [Exa Create Or Find Webset](block-integrations/exa/websets.md#exa-create-or-find-webset) | Create a new webset or return existing one by external_id (idempotent operation) |
| [Exa Create Webset](block-integrations/exa/websets.md#exa-create-webset) | Create a new Exa Webset for persistent web search collections with optional waiting for initial results |
| [Exa Create Webset Search](block-integrations/exa/websets_search.md#exa-create-webset-search) | Add a new search to an existing webset to find more items |
| [Exa Delete Enrichment](block-integrations/exa/websets_enrichment.md#exa-delete-enrichment) | Delete an enrichment from a webset |
| [Exa Delete Monitor](block-integrations/exa/websets_monitor.md#exa-delete-monitor) | Delete a monitor from a webset |
| [Exa Delete Webset](block-integrations/exa/websets.md#exa-delete-webset) | Delete a Webset and all its items |
| [Exa Delete Webset Item](block-integrations/exa/websets_items.md#exa-delete-webset-item) | Delete a specific item from a webset |
| [Exa Find Or Create Search](block-integrations/exa/websets_search.md#exa-find-or-create-search) | Find existing search by query or create new - prevents duplicate searches in workflows |
| [Exa Find Similar](block-integrations/exa/similar.md#exa-find-similar) | Finds similar links using Exa's findSimilar API |
| [Exa Get Enrichment](block-integrations/exa/websets_enrichment.md#exa-get-enrichment) | Get the status and details of a webset enrichment |
| [Exa Get Monitor](block-integrations/exa/websets_monitor.md#exa-get-monitor) | Get the details and status of a webset monitor |
| [Exa Get Research](block-integrations/exa/research.md#exa-get-research) | Get status and results of a research task |
| [Exa Get Webset](block-integrations/exa/websets.md#exa-get-webset) | Retrieve a Webset by ID or external ID |
| [Exa Get Webset Item](block-integrations/exa/websets_items.md#exa-get-webset-item) | Get a specific item from a webset by its ID |
| [Exa Get Webset Search](block-integrations/exa/websets_search.md#exa-get-webset-search) | Get the status and details of a webset search |
| [Exa List Monitors](block-integrations/exa/websets_monitor.md#exa-list-monitors) | List all monitors with optional webset filtering |
| [Exa List Research](block-integrations/exa/research.md#exa-list-research) | List all research tasks with pagination support |
| [Exa List Webset Items](block-integrations/exa/websets_items.md#exa-list-webset-items) | List items in a webset with pagination support |
| [Exa List Websets](block-integrations/exa/websets.md#exa-list-websets) | List all Websets with pagination support |
| [Exa Preview Webset](block-integrations/exa/websets.md#exa-preview-webset) | Preview how a search query will be interpreted before creating a webset |
| [Exa Search](block-integrations/exa/search.md#exa-search) | Searches the web using Exa's advanced search API |
| [Exa Update Enrichment](block-integrations/exa/websets_enrichment.md#exa-update-enrichment) | Update an existing enrichment configuration |
| [Exa Update Monitor](block-integrations/exa/websets_monitor.md#exa-update-monitor) | Update a monitor's status, schedule, or metadata |
| [Exa Update Webset](block-integrations/exa/websets.md#exa-update-webset) | Update metadata for an existing Webset |
| [Exa Wait For Enrichment](block-integrations/exa/websets_polling.md#exa-wait-for-enrichment) | Wait for a webset enrichment to complete with progress tracking |
| [Exa Wait For Research](block-integrations/exa/research.md#exa-wait-for-research) | Wait for a research task to complete with configurable timeout |
| [Exa Wait For Search](block-integrations/exa/websets_polling.md#exa-wait-for-search) | Wait for a specific webset search to complete with progress tracking |
| [Exa Wait For Webset](block-integrations/exa/websets_polling.md#exa-wait-for-webset) | Wait for a webset to reach a specific status with progress tracking |
| [Exa Webset Items Summary](block-integrations/exa/websets_items.md#exa-webset-items-summary) | Get a summary of webset items without retrieving all data |
| [Exa Webset Status](block-integrations/exa/websets.md#exa-webset-status) | Get a quick status overview of a webset |
| [Exa Webset Summary](block-integrations/exa/websets.md#exa-webset-summary) | Get a comprehensive summary of a webset with samples and statistics |
| [Extract Website Content](block-integrations/jina/search.md#extract-website-content) | This block scrapes the content from the given web URL |
| [Fact Checker](block-integrations/jina/fact_checker.md#fact-checker) | This block checks the factuality of a given statement using Jina AI's Grounding API |
| [Firecrawl Crawl](block-integrations/firecrawl/crawl.md#firecrawl-crawl) | Firecrawl crawls websites to extract comprehensive data while bypassing blockers |
| [Firecrawl Extract](block-integrations/firecrawl/extract.md#firecrawl-extract) | Firecrawl crawls websites to extract comprehensive data while bypassing blockers |
| [Firecrawl Map Website](block-integrations/firecrawl/map.md#firecrawl-map-website) | Firecrawl maps a website to extract all the links |
| [Firecrawl Scrape](block-integrations/firecrawl/scrape.md#firecrawl-scrape) | Firecrawl scrapes a website to extract comprehensive data while bypassing blockers |
| [Firecrawl Search](block-integrations/firecrawl/search.md#firecrawl-search) | Firecrawl searches the web for the given query |
| [Get Person Detail](block-integrations/apollo/person.md#get-person-detail) | Get detailed person data with Apollo API, including email reveal |
| [Get Wikipedia Summary](block-integrations/search.md#get-wikipedia-summary) | This block fetches the summary of a given topic from Wikipedia |
| [Google Maps Search](block-integrations/search.md#google-maps-search) | This block searches for local businesses using Google Maps API |
| [Search Organizations](block-integrations/apollo/organization.md#search-organizations) | Search for organizations in Apollo |
| [Search People](block-integrations/apollo/people.md#search-people) | Search for people in Apollo |
| [Search The Web](block-integrations/jina/search.md#search-the-web) | This block searches the internet for the given search query |
| [Validate Emails](block-integrations/zerobounce/validate_emails.md#validate-emails) | Validate emails |

## Social Media and Content

| Block Name | Description |
|------------|-------------|
| [Create Discord Thread](block-integrations/discord/bot_blocks.md#create-discord-thread) | Creates a new thread in a Discord channel |
| [Create Reddit Post](block-integrations/misc.md#create-reddit-post) | Create a new post on a subreddit |
| [Delete Reddit Comment](block-integrations/misc.md#delete-reddit-comment) | Delete a Reddit comment that you own |
| [Delete Reddit Post](block-integrations/misc.md#delete-reddit-post) | Delete a Reddit post that you own |
| [Delete Telegram Message](block-integrations/telegram/blocks.md#delete-telegram-message) | Delete a message from a Telegram chat |
| [Discord Channel Info](block-integrations/discord/bot_blocks.md#discord-channel-info) | Resolves Discord channel names to IDs and vice versa |
| [Discord Get Current User](block-integrations/discord/oauth_blocks.md#discord-get-current-user) | Gets information about the currently authenticated Discord user using OAuth2 credentials |
| [Discord User Info](block-integrations/discord/bot_blocks.md#discord-user-info) | Gets information about a Discord user by their ID |
| [Edit Reddit Post](block-integrations/misc.md#edit-reddit-post) | Edit the body text of an existing Reddit post that you own |
| [Edit Telegram Message](block-integrations/telegram/blocks.md#edit-telegram-message) | Edit the text of an existing message sent by the bot |
| [Get Linkedin Profile](block-integrations/enrichlayer/linkedin.md#get-linkedin-profile) | Fetch LinkedIn profile data using Enrichlayer |
| [Get Linkedin Profile Picture](block-integrations/enrichlayer/linkedin.md#get-linkedin-profile-picture) | Get LinkedIn profile pictures using Enrichlayer |
| [Get Reddit Comment](block-integrations/misc.md#get-reddit-comment) | Get details about a specific Reddit comment by its ID |
| [Get Reddit Comment Replies](block-integrations/misc.md#get-reddit-comment-replies) | Get replies to a specific Reddit comment |
| [Get Reddit Inbox](block-integrations/misc.md#get-reddit-inbox) | Get messages, mentions, and comment replies from your Reddit inbox |
| [Get Reddit Post](block-integrations/misc.md#get-reddit-post) | Get detailed information about a specific Reddit post by its ID |
| [Get Reddit Post Comments](block-integrations/misc.md#get-reddit-post-comments) | Get top-level comments on a Reddit post |
| [Get Reddit Posts](block-integrations/misc.md#get-reddit-posts) | This block fetches Reddit posts from a defined subreddit name |
| [Get Reddit User Info](block-integrations/misc.md#get-reddit-user-info) | Get information about a Reddit user including karma, account age, and verification status |
| [Get Subreddit Flairs](block-integrations/misc.md#get-subreddit-flairs) | Get available link flair options for a subreddit |
| [Get Subreddit Info](block-integrations/misc.md#get-subreddit-info) | Get information about a subreddit including subscriber count, description, and rules |
| [Get Subreddit Rules](block-integrations/misc.md#get-subreddit-rules) | Get the rules for a subreddit to ensure compliance before posting |
| [Get Telegram File](block-integrations/telegram/blocks.md#get-telegram-file) | Download a file from Telegram using its file_id |
| [Get User Posts](block-integrations/misc.md#get-user-posts) | Fetch posts by a specific Reddit user |
| [Linkedin Person Lookup](block-integrations/enrichlayer/linkedin.md#linkedin-person-lookup) | Look up LinkedIn profiles by person information using Enrichlayer |
| [Linkedin Role Lookup](block-integrations/enrichlayer/linkedin.md#linkedin-role-lookup) | Look up LinkedIn profiles by role in a company using Enrichlayer |
| [Post Reddit Comment](block-integrations/misc.md#post-reddit-comment) | This block posts a Reddit comment on a specified Reddit post |
| [Post To Bluesky](block-integrations/ayrshare/post_to_bluesky.md#post-to-bluesky) | Post to Bluesky using Ayrshare |
| [Post To Facebook](block-integrations/ayrshare/post_to_facebook.md#post-to-facebook) | Post to Facebook using Ayrshare |
| [Post To GMB](block-integrations/ayrshare/post_to_gmb.md#post-to-gmb) | Post to Google My Business using Ayrshare |
| [Post To Instagram](block-integrations/ayrshare/post_to_instagram.md#post-to-instagram) | Post to Instagram using Ayrshare |
| [Post To Linked In](block-integrations/ayrshare/post_to_linkedin.md#post-to-linked-in) | Post to LinkedIn using Ayrshare |
| [Post To Pinterest](block-integrations/ayrshare/post_to_pinterest.md#post-to-pinterest) | Post to Pinterest using Ayrshare |
| [Post To Reddit](block-integrations/ayrshare/post_to_reddit.md#post-to-reddit) | Post to Reddit using Ayrshare |
| [Post To Snapchat](block-integrations/ayrshare/post_to_snapchat.md#post-to-snapchat) | Post to Snapchat using Ayrshare |
| [Post To Telegram](block-integrations/ayrshare/post_to_telegram.md#post-to-telegram) | Post to Telegram using Ayrshare |
| [Post To Threads](block-integrations/ayrshare/post_to_threads.md#post-to-threads) | Post to Threads using Ayrshare |
| [Post To Tik Tok](block-integrations/ayrshare/post_to_tiktok.md#post-to-tik-tok) | Post to TikTok using Ayrshare |
| [Post To X](block-integrations/ayrshare/post_to_x.md#post-to-x) | Post to X / Twitter using Ayrshare |
| [Post To You Tube](block-integrations/ayrshare/post_to_youtube.md#post-to-you-tube) | Post to YouTube using Ayrshare |
| [Publish To Medium](block-integrations/misc.md#publish-to-medium) | Publishes a post to Medium |
| [Read Discord Messages](block-integrations/discord/bot_blocks.md#read-discord-messages) | Reads messages from a Discord channel using a bot token |
| [Reddit Get My Posts](block-integrations/misc.md#reddit-get-my-posts) | Fetch posts created by the authenticated Reddit user (you) |
| [Reply To Discord Message](block-integrations/discord/bot_blocks.md#reply-to-discord-message) | Replies to a specific Discord message |
| [Reply To Reddit Comment](block-integrations/misc.md#reply-to-reddit-comment) | Reply to a specific Reddit comment |
| [Reply To Telegram Message](block-integrations/telegram/blocks.md#reply-to-telegram-message) | Reply to a specific message in a Telegram chat |
| [Search Reddit](block-integrations/misc.md#search-reddit) | Search Reddit for posts matching a query |
| [Send Discord DM](block-integrations/discord/bot_blocks.md#send-discord-dm) | Sends a direct message to a Discord user using their user ID |
| [Send Discord Embed](block-integrations/discord/bot_blocks.md#send-discord-embed) | Sends a rich embed message to a Discord channel |
| [Send Discord File](block-integrations/discord/bot_blocks.md#send-discord-file) | Sends a file attachment to a Discord channel |
| [Send Discord Message](block-integrations/discord/bot_blocks.md#send-discord-message) | Sends a message to a Discord channel using a bot token |
| [Send Reddit Message](block-integrations/misc.md#send-reddit-message) | Send a private message (DM) to a Reddit user |
| [Send Telegram Audio](block-integrations/telegram/blocks.md#send-telegram-audio) | Send an audio file to a Telegram chat |
| [Send Telegram Document](block-integrations/telegram/blocks.md#send-telegram-document) | Send a document (any file type) to a Telegram chat |
| [Send Telegram Message](block-integrations/telegram/blocks.md#send-telegram-message) | Send a text message to a Telegram chat |
| [Send Telegram Photo](block-integrations/telegram/blocks.md#send-telegram-photo) | Send a photo to a Telegram chat |
| [Send Telegram Video](block-integrations/telegram/blocks.md#send-telegram-video) | Send a video to a Telegram chat |
| [Send Telegram Voice](block-integrations/telegram/blocks.md#send-telegram-voice) | Send a voice message to a Telegram chat |
| [Telegram Message Reaction Trigger](block-integrations/telegram/triggers.md#telegram-message-reaction-trigger) | Triggers when a reaction to a message is changed |
| [Telegram Message Trigger](block-integrations/telegram/triggers.md#telegram-message-trigger) | Triggers when a message is received or edited in your Telegram bot |
| [Transcribe Youtube Video](block-integrations/misc.md#transcribe-youtube-video) | Transcribes a YouTube video using a proxy |
| [Twitter Add List Member](block-integrations/twitter/list_members.md#twitter-add-list-member) | This block adds a specified user to a Twitter List owned by the authenticated user |
| [Twitter Bookmark Tweet](block-integrations/twitter/bookmark.md#twitter-bookmark-tweet) | This block bookmarks a tweet on Twitter |
| [Twitter Create List](block-integrations/twitter/manage_lists.md#twitter-create-list) | This block creates a new Twitter List for the authenticated user |
| [Twitter Delete List](block-integrations/twitter/manage_lists.md#twitter-delete-list) | This block deletes a specified Twitter List owned by the authenticated user |
| [Twitter Delete Tweet](block-integrations/twitter/manage.md#twitter-delete-tweet) | This block deletes a tweet on Twitter |
| [Twitter Follow List](block-integrations/twitter/list_follows.md#twitter-follow-list) | This block follows a specified Twitter list for the authenticated user |
| [Twitter Follow User](block-integrations/twitter/follows.md#twitter-follow-user) | This block follows a specified Twitter user |
| [Twitter Get Blocked Users](block-integrations/twitter/blocks.md#twitter-get-blocked-users) | This block retrieves a list of users blocked by the authenticating user |
| [Twitter Get Bookmarked Tweets](block-integrations/twitter/bookmark.md#twitter-get-bookmarked-tweets) | This block retrieves bookmarked tweets from Twitter |
| [Twitter Get Followers](block-integrations/twitter/follows.md#twitter-get-followers) | This block retrieves followers of a specified Twitter user |
| [Twitter Get Following](block-integrations/twitter/follows.md#twitter-get-following) | This block retrieves the users that a specified Twitter user is following |
| [Twitter Get Home Timeline](block-integrations/twitter/timeline.md#twitter-get-home-timeline) | This block retrieves the authenticated user's home timeline |
| [Twitter Get Liked Tweets](block-integrations/twitter/like.md#twitter-get-liked-tweets) | This block gets information about tweets liked by a user |
| [Twitter Get Liking Users](block-integrations/twitter/like.md#twitter-get-liking-users) | This block gets information about users who liked a tweet |
| [Twitter Get List](block-integrations/twitter/list_lookup.md#twitter-get-list) | This block retrieves information about a specified Twitter List |
| [Twitter Get List Members](block-integrations/twitter/list_members.md#twitter-get-list-members) | This block retrieves the members of a specified Twitter List |
| [Twitter Get List Memberships](block-integrations/twitter/list_members.md#twitter-get-list-memberships) | This block retrieves all Lists that a specified user is a member of |
| [Twitter Get List Tweets](block-integrations/twitter/list_tweets_lookup.md#twitter-get-list-tweets) | This block retrieves tweets from a specified Twitter list |
| [Twitter Get Muted Users](block-integrations/twitter/mutes.md#twitter-get-muted-users) | This block gets a list of users muted by the authenticating user |
| [Twitter Get Owned Lists](block-integrations/twitter/list_lookup.md#twitter-get-owned-lists) | This block retrieves all Lists owned by a specified Twitter user |
| [Twitter Get Pinned Lists](block-integrations/twitter/pinned_lists.md#twitter-get-pinned-lists) | This block returns the Lists pinned by the authenticated user |
| [Twitter Get Quote Tweets](block-integrations/twitter/quote.md#twitter-get-quote-tweets) | This block gets quote tweets for a specific tweet |
| [Twitter Get Retweeters](block-integrations/twitter/retweet.md#twitter-get-retweeters) | This block gets information about who has retweeted a tweet |
| [Twitter Get Space Buyers](block-integrations/twitter/spaces_lookup.md#twitter-get-space-buyers) | This block retrieves a list of users who purchased tickets to a Twitter Space |
| [Twitter Get Space By Id](block-integrations/twitter/spaces_lookup.md#twitter-get-space-by-id) | This block retrieves information about a single Twitter Space |
| [Twitter Get Space Tweets](block-integrations/twitter/spaces_lookup.md#twitter-get-space-tweets) | This block retrieves tweets shared in a Twitter Space |
| [Twitter Get Spaces](block-integrations/twitter/spaces_lookup.md#twitter-get-spaces) | This block retrieves information about multiple Twitter Spaces |
| [Twitter Get Tweet](block-integrations/twitter/tweet_lookup.md#twitter-get-tweet) | This block retrieves information about a specific Tweet |
| [Twitter Get Tweets](block-integrations/twitter/tweet_lookup.md#twitter-get-tweets) | This block retrieves information about multiple Tweets |
| [Twitter Get User](block-integrations/twitter/user_lookup.md#twitter-get-user) | This block retrieves information about a specified Twitter user |
| [Twitter Get User Mentions](block-integrations/twitter/timeline.md#twitter-get-user-mentions) | This block retrieves Tweets mentioning a specific user |
| [Twitter Get User Tweets](block-integrations/twitter/timeline.md#twitter-get-user-tweets) | This block retrieves Tweets composed by a single user |
| [Twitter Get Users](block-integrations/twitter/user_lookup.md#twitter-get-users) | This block retrieves information about multiple Twitter users |
| [Twitter Hide Reply](block-integrations/twitter/hide.md#twitter-hide-reply) | This block hides a reply to a tweet |
| [Twitter Like Tweet](block-integrations/twitter/like.md#twitter-like-tweet) | This block likes a tweet |
| [Twitter Mute User](block-integrations/twitter/mutes.md#twitter-mute-user) | This block mutes a specified Twitter user |
| [Twitter Pin List](block-integrations/twitter/pinned_lists.md#twitter-pin-list) | This block allows the authenticated user to pin a specified List |
| [Twitter Post Tweet](block-integrations/twitter/manage.md#twitter-post-tweet) | This block posts a tweet on Twitter |
| [Twitter Remove Bookmark Tweet](block-integrations/twitter/bookmark.md#twitter-remove-bookmark-tweet) | This block removes a bookmark from a tweet on Twitter |
| [Twitter Remove List Member](block-integrations/twitter/list_members.md#twitter-remove-list-member) | This block removes a specified user from a Twitter List owned by the authenticated user |
| [Twitter Remove Retweet](block-integrations/twitter/retweet.md#twitter-remove-retweet) | This block removes a retweet on Twitter |
| [Twitter Retweet](block-integrations/twitter/retweet.md#twitter-retweet) | This block retweets a tweet on Twitter |
| [Twitter Search Recent Tweets](block-integrations/twitter/manage.md#twitter-search-recent-tweets) | This block searches all public Tweets in Twitter history |
| [Twitter Search Spaces](block-integrations/twitter/search_spaces.md#twitter-search-spaces) | This block searches for Twitter Spaces based on specified terms |
| [Twitter Unfollow List](block-integrations/twitter/list_follows.md#twitter-unfollow-list) | This block unfollows a specified Twitter list for the authenticated user |
| [Twitter Unfollow User](block-integrations/twitter/follows.md#twitter-unfollow-user) | This block unfollows a specified Twitter user |
| [Twitter Unhide Reply](block-integrations/twitter/hide.md#twitter-unhide-reply) | This block unhides a reply to a tweet |
| [Twitter Unlike Tweet](block-integrations/twitter/like.md#twitter-unlike-tweet) | This block unlikes a tweet |
| [Twitter Unmute User](block-integrations/twitter/mutes.md#twitter-unmute-user) | This block unmutes a specified Twitter user |
| [Twitter Unpin List](block-integrations/twitter/pinned_lists.md#twitter-unpin-list) | This block allows the authenticated user to unpin a specified List |
| [Twitter Update List](block-integrations/twitter/manage_lists.md#twitter-update-list) | This block updates a specified Twitter List owned by the authenticated user |

## Communication

| Block Name | Description |
|------------|-------------|
| [Baas Bot Join Meeting](block-integrations/baas/bots.md#baas-bot-join-meeting) | Deploy a bot to join and record a meeting |
| [Baas Bot Leave Meeting](block-integrations/baas/bots.md#baas-bot-leave-meeting) | Remove a bot from an ongoing meeting |
| [Gmail Add Label](block-integrations/google/gmail.md#gmail-add-label) | A block that adds a label to a specific email message in Gmail, creating the label if it doesn't exist |
| [Gmail Create Draft](block-integrations/google/gmail.md#gmail-create-draft) | Create draft emails in Gmail with automatic HTML detection and proper text formatting |
| [Gmail Draft Reply](block-integrations/google/gmail.md#gmail-draft-reply) | Create draft replies to Gmail threads with automatic HTML detection and proper text formatting |
| [Gmail Forward](block-integrations/google/gmail.md#gmail-forward) | Forward Gmail messages to other recipients with automatic HTML detection and proper formatting |
| [Gmail Get Profile](block-integrations/google/gmail.md#gmail-get-profile) | Get the authenticated user's Gmail profile details including email address and message statistics |
| [Gmail Get Thread](block-integrations/google/gmail.md#gmail-get-thread) | A block that retrieves an entire Gmail thread (email conversation) by ID, returning all messages with decoded bodies for reading complete conversations |
| [Gmail List Labels](block-integrations/google/gmail.md#gmail-list-labels) | A block that retrieves all labels (categories) from a Gmail account for organizing and categorizing emails |
| [Gmail Read](block-integrations/google/gmail.md#gmail-read) | A block that retrieves and reads emails from a Gmail account based on search criteria, returning detailed message information including subject, sender, body, and attachments |
| [Gmail Remove Label](block-integrations/google/gmail.md#gmail-remove-label) | A block that removes a label from a specific email message in a Gmail account |
| [Gmail Reply](block-integrations/google/gmail.md#gmail-reply) | Reply to Gmail threads with automatic HTML detection and proper text formatting |
| [Gmail Send](block-integrations/google/gmail.md#gmail-send) | Send emails via Gmail with automatic HTML detection and proper text formatting |
| [Hub Spot Engagement](block-integrations/hubspot/engagement.md#hub-spot-engagement) | Manages HubSpot engagements - sends emails and tracks engagement metrics |

## Developer Tools

| Block Name | Description |
|------------|-------------|
| [Exa Code Context](block-integrations/exa/code_context.md#exa-code-context) | Search billions of GitHub repos, docs, and Stack Overflow for relevant code examples |
| [Execute Code](block-integrations/misc.md#execute-code) | Executes code in a sandbox environment with internet access |
| [Execute Code Step](block-integrations/misc.md#execute-code-step) | Execute code in a previously instantiated sandbox |
| [Github Add Label](block-integrations/github/issues.md#github-add-label) | A block that adds a label to a GitHub issue or pull request for categorization and organization |
| [Github Assign Issue](block-integrations/github/issues.md#github-assign-issue) | A block that assigns a GitHub user to an issue for task ownership and tracking |
| [Github Assign PR Reviewer](block-integrations/github/pull_requests.md#github-assign-pr-reviewer) | This block assigns a reviewer to a specified GitHub pull request |
| [Github Comment](block-integrations/github/issues.md#github-comment) | A block that posts comments on GitHub issues or pull requests using the GitHub API |
| [Github Create Check Run](block-integrations/github/checks.md#github-create-check-run) | Creates a new check run for a specific commit in a GitHub repository |
| [Github Create Comment Object](block-integrations/github/reviews.md#github-create-comment-object) | Creates a comment object for use with GitHub blocks |
| [Github Create File](block-integrations/github/repo.md#github-create-file) | This block creates a new file in a GitHub repository |
| [Github Create PR Review](block-integrations/github/reviews.md#github-create-pr-review) | This block creates a review on a GitHub pull request with optional inline comments |
| [Github Create Repository](block-integrations/github/repo.md#github-create-repository) | This block creates a new GitHub repository |
| [Github Create Status](block-integrations/github/statuses.md#github-create-status) | Creates a new commit status in a GitHub repository |
| [Github Delete Branch](block-integrations/github/repo.md#github-delete-branch) | This block deletes a specified branch |
| [Github Discussion Trigger](block-integrations/github/triggers.md#github-discussion-trigger) | This block triggers on GitHub Discussions events |
| [Github Get CI Results](block-integrations/github/ci.md#github-get-ci-results) | This block gets CI results for a commit or PR, with optional search for specific errors/warnings in logs |
| [Github Get PR Review Comments](block-integrations/github/reviews.md#github-get-pr-review-comments) | This block gets all review comments from a GitHub pull request or from a specific review |
| [Github Issues Trigger](block-integrations/github/triggers.md#github-issues-trigger) | This block triggers on GitHub issues events |
| [Github List Branches](block-integrations/github/repo.md#github-list-branches) | This block lists all branches for a specified GitHub repository |
| [Github List Comments](block-integrations/github/issues.md#github-list-comments) | A block that retrieves all comments from a GitHub issue or pull request, including comment metadata and content |
| [Github List Discussions](block-integrations/github/repo.md#github-list-discussions) | This block lists recent discussions for a specified GitHub repository |
| [Github List Issues](block-integrations/github/issues.md#github-list-issues) | A block that retrieves a list of issues from a GitHub repository with their titles and URLs |
| [Github List PR Reviewers](block-integrations/github/pull_requests.md#github-list-pr-reviewers) | This block lists all reviewers for a specified GitHub pull request |
| [Github List PR Reviews](block-integrations/github/reviews.md#github-list-pr-reviews) | This block lists all reviews for a specified GitHub pull request |
| [Github List Pull Requests](block-integrations/github/pull_requests.md#github-list-pull-requests) | This block lists all pull requests for a specified GitHub repository |
| [Github List Releases](block-integrations/github/repo.md#github-list-releases) | This block lists all releases for a specified GitHub repository |
| [Github List Stargazers](block-integrations/github/repo.md#github-list-stargazers) | This block lists all users who have starred a specified GitHub repository |
| [Github List Tags](block-integrations/github/repo.md#github-list-tags) | This block lists all tags for a specified GitHub repository |
| [Github Make Branch](block-integrations/github/repo.md#github-make-branch) | This block creates a new branch from a specified source branch |
| [Github Make Issue](block-integrations/github/issues.md#github-make-issue) | A block that creates new issues on GitHub repositories with a title and body content |
| [Github Make Pull Request](block-integrations/github/pull_requests.md#github-make-pull-request) | This block creates a new pull request on a specified GitHub repository |
| [Github Pull Request Trigger](block-integrations/github/triggers.md#github-pull-request-trigger) | This block triggers on pull request events and outputs the event type and payload |
| [Github Read File](block-integrations/github/repo.md#github-read-file) | This block reads the content of a specified file from a GitHub repository |
| [Github Read Folder](block-integrations/github/repo.md#github-read-folder) | This block reads the content of a specified folder from a GitHub repository |
| [Github Read Issue](block-integrations/github/issues.md#github-read-issue) | A block that retrieves information about a specific GitHub issue, including its title, body content, and creator |
| [Github Read Pull Request](block-integrations/github/pull_requests.md#github-read-pull-request) | This block reads the body, title, user, and changes of a specified GitHub pull request |
| [Github Release Trigger](block-integrations/github/triggers.md#github-release-trigger) | This block triggers on GitHub release events |
| [Github Remove Label](block-integrations/github/issues.md#github-remove-label) | A block that removes a label from a GitHub issue or pull request |
| [Github Resolve Review Discussion](block-integrations/github/reviews.md#github-resolve-review-discussion) | This block resolves or unresolves a review discussion thread on a GitHub pull request |
| [Github Star Trigger](block-integrations/github/triggers.md#github-star-trigger) | This block triggers on GitHub star events |
| [Github Submit Pending Review](block-integrations/github/reviews.md#github-submit-pending-review) | This block submits a pending (draft) review on a GitHub pull request |
| [Github Unassign Issue](block-integrations/github/issues.md#github-unassign-issue) | A block that removes a user's assignment from a GitHub issue |
| [Github Unassign PR Reviewer](block-integrations/github/pull_requests.md#github-unassign-pr-reviewer) | This block unassigns a reviewer from a specified GitHub pull request |
| [Github Update Check Run](block-integrations/github/checks.md#github-update-check-run) | Updates an existing check run in a GitHub repository |
| [Github Update Comment](block-integrations/github/issues.md#github-update-comment) | A block that updates an existing comment on a GitHub issue or pull request |
| [Github Update File](block-integrations/github/repo.md#github-update-file) | This block updates an existing file in a GitHub repository |
| [Instantiate Code Sandbox](block-integrations/misc.md#instantiate-code-sandbox) | Instantiate a sandbox environment with internet access in which you can execute code with the Execute Code Step block |
| [MCP Tool](block-integrations/mcp/block.md#mcp-tool) | Connect to any MCP server and execute its tools |
| [Slant3D Order Webhook](block-integrations/slant3d/webhook.md#slant3d-order-webhook) | This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped |

## Media Generation

| Block Name | Description |
|------------|-------------|
| [Add Audio To Video](block-integrations/video/add_audio.md#add-audio-to-video) | Block to attach an audio file to a video file using moviepy |
| [Loop Video](block-integrations/video/loop.md#loop-video) | Block to loop a video to a given duration or number of repeats |
| [Media Duration](block-integrations/video/duration.md#media-duration) | Block to get the duration of a media file |
| [Video Clip](block-integrations/video/clip.md#video-clip) | Extract a time segment from a video |
| [Video Concat](block-integrations/video/concat.md#video-concat) | Merge multiple video clips into one continuous video |
| [Video Download](block-integrations/video/download.md#video-download) | Download video from URL (YouTube, Vimeo, news sites, direct links) |
| [Video Text Overlay](block-integrations/video/text_overlay.md#video-text-overlay) | Add text overlay/caption to video |

## Productivity

| Block Name | Description |
|------------|-------------|
| [Google Calendar Create Event](block-integrations/google/calendar.md#google-calendar-create-event) | This block creates a new event in Google Calendar with customizable parameters |
| [Notion Create Page](block-integrations/notion/create_page.md#notion-create-page) | Create a new page in Notion |
| [Notion Read Database](block-integrations/notion/read_database.md#notion-read-database) | Query a Notion database with optional filtering and sorting, returning structured entries |
| [Notion Read Page](block-integrations/notion/read_page.md#notion-read-page) | Read a Notion page by its ID and return its raw JSON |
| [Notion Read Page Markdown](block-integrations/notion/read_page_markdown.md#notion-read-page-markdown) | Read a Notion page and convert it to Markdown format with proper formatting for headings, lists, links, and rich text |
| [Notion Search](block-integrations/notion/search.md#notion-search) | Search your Notion workspace for pages and databases by text query |
| [Todoist Close Task](block-integrations/todoist/tasks.md#todoist-close-task) | Closes a task in Todoist |
| [Todoist Create Comment](block-integrations/todoist/comments.md#todoist-create-comment) | Creates a new comment on a Todoist task or project |
| [Todoist Create Label](block-integrations/todoist/labels.md#todoist-create-label) | Creates a new label in Todoist, It will not work if same name already exists |
| [Todoist Create Project](block-integrations/todoist/projects.md#todoist-create-project) | Creates a new project in Todoist |
| [Todoist Create Task](block-integrations/todoist/tasks.md#todoist-create-task) | Creates a new task in a Todoist project |
| [Todoist Delete Comment](block-integrations/todoist/comments.md#todoist-delete-comment) | Deletes a Todoist comment |
| [Todoist Delete Label](block-integrations/todoist/labels.md#todoist-delete-label) | Deletes a personal label in Todoist |
| [Todoist Delete Project](block-integrations/todoist/projects.md#todoist-delete-project) | Deletes a Todoist project and all its contents |
| [Todoist Delete Section](block-integrations/todoist/sections.md#todoist-delete-section) | Deletes a section and all its tasks from Todoist |
| [Todoist Delete Task](block-integrations/todoist/tasks.md#todoist-delete-task) | Deletes a task in Todoist |
| [Todoist Get Comment](block-integrations/todoist/comments.md#todoist-get-comment) | Get a single comment from Todoist |
| [Todoist Get Comments](block-integrations/todoist/comments.md#todoist-get-comments) | Get all comments for a Todoist task or project |
| [Todoist Get Label](block-integrations/todoist/labels.md#todoist-get-label) | Gets a personal label from Todoist by ID |
| [Todoist Get Project](block-integrations/todoist/projects.md#todoist-get-project) | Gets details for a specific Todoist project |
| [Todoist Get Section](block-integrations/todoist/sections.md#todoist-get-section) | Gets a single section by ID from Todoist |
| [Todoist Get Shared Labels](block-integrations/todoist/labels.md#todoist-get-shared-labels) | Gets all shared labels from Todoist |
| [Todoist Get Task](block-integrations/todoist/tasks.md#todoist-get-task) | Get an active task from Todoist |
| [Todoist Get Tasks](block-integrations/todoist/tasks.md#todoist-get-tasks) | Get active tasks from Todoist |
| [Todoist List Collaborators](block-integrations/todoist/projects.md#todoist-list-collaborators) | Gets all collaborators for a specific Todoist project |
| [Todoist List Labels](block-integrations/todoist/labels.md#todoist-list-labels) | Gets all personal labels from Todoist |
| [Todoist List Projects](block-integrations/todoist/projects.md#todoist-list-projects) | Gets all projects and their details from Todoist |
| [Todoist List Sections](block-integrations/todoist/sections.md#todoist-list-sections) | Gets all sections and their details from Todoist |
| [Todoist Remove Shared Labels](block-integrations/todoist/labels.md#todoist-remove-shared-labels) | Removes all instances of a shared label |
| [Todoist Rename Shared Labels](block-integrations/todoist/labels.md#todoist-rename-shared-labels) | Renames all instances of a shared label |
| [Todoist Reopen Task](block-integrations/todoist/tasks.md#todoist-reopen-task) | Reopens a task in Todoist |
| [Todoist Update Comment](block-integrations/todoist/comments.md#todoist-update-comment) | Updates a Todoist comment |
| [Todoist Update Label](block-integrations/todoist/labels.md#todoist-update-label) | Updates a personal label in Todoist |
| [Todoist Update Project](block-integrations/todoist/projects.md#todoist-update-project) | Updates an existing project in Todoist |
| [Todoist Update Task](block-integrations/todoist/tasks.md#todoist-update-task) | Updates an existing task in Todoist |

## Logic and Control Flow

| Block Name | Description |
|------------|-------------|
| [Calculator](block-integrations/logic.md#calculator) | Performs a mathematical operation on two numbers |
| [Condition](block-integrations/logic.md#condition) | Handles conditional logic based on comparison operators |
| [Count Items](block-integrations/logic.md#count-items) | Counts the number of items in a collection |
| [Data Sampling](block-integrations/logic.md#data-sampling) | This block samples data from a given dataset using various sampling methods |
| [Exa Webset Ready Check](block-integrations/exa/websets.md#exa-webset-ready-check) | Check if webset is ready for next operation - enables conditional workflow branching |
| [If Input Matches](block-integrations/logic.md#if-input-matches) | Handles conditional logic based on comparison operators |
| [Pinecone Init](block-integrations/logic.md#pinecone-init) | Initializes a Pinecone index |
| [Pinecone Insert](block-integrations/logic.md#pinecone-insert) | Upload data to a Pinecone index |
| [Pinecone Query](block-integrations/logic.md#pinecone-query) | Queries a Pinecone index |
| [Step Through Items](block-integrations/logic.md#step-through-items) | Iterates over a list or dictionary and outputs each item |

## Input/Output

| Block Name | Description |
|------------|-------------|
| [Exa Webset Webhook](block-integrations/exa/webhook_blocks.md#exa-webset-webhook) | Receive webhook notifications for Exa webset events |
| [Generic Webhook Trigger](block-integrations/generic_webhook/triggers.md#generic-webhook-trigger) | This block will output the contents of the generic input for the webhook |
| [Read RSS Feed](block-integrations/misc.md#read-rss-feed) | Reads RSS feed entries from a given URL |
| [Send Authenticated Web Request](block-integrations/misc.md#send-authenticated-web-request) | Make an authenticated HTTP request with host-scoped credentials (JSON / form / multipart) |
| [Send Email](block-integrations/misc.md#send-email) | This block sends an email using the provided SMTP credentials |
| [Send Web Request](block-integrations/misc.md#send-web-request) | Make an HTTP request (JSON / form / multipart) |

## Agent Integration

| Block Name | Description |
|------------|-------------|
| [Agent Executor](block-integrations/misc.md#agent-executor) | Executes an existing agent inside your agent |

## CRM Services

| Block Name | Description |
|------------|-------------|
| [Add Lead To Campaign](block-integrations/smartlead/campaign.md#add-lead-to-campaign) | Add a lead to a campaign in SmartLead |
| [Create Campaign](block-integrations/smartlead/campaign.md#create-campaign) | Create a campaign in SmartLead |
| [Hub Spot Company](block-integrations/hubspot/company.md#hub-spot-company) | Manages HubSpot companies - create, update, and retrieve company information |
| [Hub Spot Contact](block-integrations/hubspot/contact.md#hub-spot-contact) | Manages HubSpot contacts - create, update, and retrieve contact information |
| [Save Campaign Sequences](block-integrations/smartlead/campaign.md#save-campaign-sequences) | Save sequences within a campaign |

## AI Safety

| Block Name | Description |
|------------|-------------|
| [Nvidia Deepfake Detect](block-integrations/nvidia/deepfake.md#nvidia-deepfake-detect) | Detects potential deepfakes in images using Nvidia's AI API |

## Issue Tracking

| Block Name | Description |
|------------|-------------|
| [Linear Create Comment](block-integrations/linear/comment.md#linear-create-comment) | Creates a new comment on a Linear issue |
| [Linear Create Issue](block-integrations/linear/issues.md#linear-create-issue) | Creates a new issue on Linear |
| [Linear Get Project Issues](block-integrations/linear/issues.md#linear-get-project-issues) | Gets issues from a Linear project filtered by status and assignee |
| [Linear Search Issues](block-integrations/linear/issues.md#linear-search-issues) | Searches for issues on Linear |
| [Linear Search Projects](block-integrations/linear/projects.md#linear-search-projects) | Searches for projects on Linear |

## Hardware

| Block Name | Description |
|------------|-------------|
| [Compass AI Trigger](block-integrations/compass/triggers.md#compass-ai-trigger) | This block will output the contents of the compass transcription |
