# AutoGPT Blocks Overview

AutoGPT uses a modular approach with various "blocks" to handle different tasks. These blocks are the building blocks of AutoGPT workflows, allowing users to create complex automations by combining simple, specialized components.

Below is a comprehensive list of all available blocks, categorized by their primary function. Click on any block name to view its detailed documentation.

## Basic Operations
| Block Name | Description |
|------------|-------------|
| [Store Value](basic.md#store-value) | Stores and forwards a value |
| [Print to Console](basic.md#print-to-console) | Outputs text to the console for debugging |
| [Find in Dictionary](basic.md#find-in-dictionary) | Looks up a value in a dictionary or list |
| [Agent Input](basic.md#agent-input) | Accepts user input in a workflow |
| [Agent Output](basic.md#agent-output) | Records and formats workflow results |
| [Add to Dictionary](basic.md#add-to-dictionary) | Adds a new key-value pair to a dictionary |
| [Add to List](basic.md#add-to-list) | Adds a new entry to a list |
| [Note](basic.md#note) | Displays a sticky note in the workflow |

## Data Processing
| Block Name | Description |
|------------|-------------|
| [Read CSV](csv.md#read-csv) | Processes and extracts data from CSV files |
| [Data Sampling](sampling.md#data-sampling) | Selects a subset of data using various sampling methods |

## Text Processing
| Block Name | Description |
|------------|-------------|
| [Match Text Pattern](text.md#match-text-pattern) | Checks if text matches a specified pattern |
| [Extract Text Information](text.md#extract-text-information) | Extracts specific information from text using patterns |
| [Fill Text Template](text.md#fill-text-template) | Populates a template with provided values |
| [Combine Texts](text.md#combine-texts) | Merges multiple text inputs into one |
| [Text Decoder](decoder_block.md#text-decoder) | Converts encoded text into readable format |

## AI and Language Models
| Block Name | Description |
|------------|-------------|
| [AI Structured Response Generator](llm.md#ai-structured-response-generator) | Generates structured responses using LLMs |
| [AI Text Generator](llm.md#ai-text-generator) | Produces text responses using LLMs |
| [AI Text Summarizer](llm.md#ai-text-summarizer) | Summarizes long texts using LLMs |
| [AI Conversation](llm.md#ai-conversation) | Facilitates multi-turn conversations with LLMs |
| [AI List Generator](llm.md#ai-list-generator) | Creates lists based on prompts using LLMs |

## Web and API Interactions
| Block Name | Description |
|------------|-------------|
| [Send Web Request](http.md#send-web-request) | Makes HTTP requests to specified web addresses |
| [Read RSS Feed](rss.md#read-rss-feed) | Retrieves and processes entries from RSS feeds |
| [Get Weather Information](search.md#get-weather-information) | Fetches current weather data for a location |
| [Google Maps Search](google_maps.md#google-maps-search) | Searches for local businesses using Google Maps API |

## Social Media and Content
| Block Name | Description |
|------------|-------------|
| [Get Reddit Posts](reddit.md#get-reddit-posts) | Retrieves posts from specified subreddits |
| [Post Reddit Comment](reddit.md#post-reddit-comment) | Posts comments on Reddit |
| [Publish to Medium](medium.md#publish-to-medium) | Publishes content directly to Medium |
| [Read Discord Messages](discord.md#read-discord-messages) | Retrieves messages from Discord channels |
| [Send Discord Message](discord.md#send-discord-message) | Sends messages to Discord channels |

## Search and Information Retrieval
| Block Name | Description |
|------------|-------------|
| [Get Wikipedia Summary](search.md#get-wikipedia-summary) | Fetches summaries of topics from Wikipedia |
| [Search The Web](search.md#search-the-web) | Performs web searches and returns results |
| [Extract Website Content](search.md#extract-website-content) | Retrieves and extracts content from websites |

## Time and Date
| Block Name | Description |
|------------|-------------|
| [Get Current Time](time_blocks.md#get-current-time) | Provides the current time |
| [Get Current Date](time_blocks.md#get-current-date) | Provides the current date |
| [Get Current Date and Time](time_blocks.md#get-current-date-and-time) | Provides both current date and time |
| [Countdown Timer](time_blocks.md#countdown-timer) | Acts as a countdown timer |

## Math and Calculations
| Block Name | Description |
|------------|-------------|
| [Calculator](maths.md#calculator) | Performs basic mathematical operations |
| [Count Items](maths.md#count-items) | Counts items in a collection |

## Media Generation
| Block Name | Description |
|------------|-------------|
| [Ideogram Model](ideogram.md#ideogram-model) | Generates images based on text prompts |
| [Create Talking Avatar Video](talking_head.md#create-talking-avatar-video) | Creates videos with talking avatars |
| [Unreal Text to Speech](text_to_speech_block.md#unreal-text-to-speech) | Converts text to speech using Unreal Speech API |
| [AI Shortform Video Creator](ai_shortform_video_block.md#ai-shortform-video-creator) | Generates short-form videos using AI |
| [Replicate Flux Advanced Model](replicate_flux_advanced.md#replicate-flux-advanced-model) | Creates images using Replicate's Flux models |

## Miscellaneous
| Block Name | Description |
|------------|-------------|
| [Transcribe YouTube Video](youtube.md#transcribe-youtube-video) | Transcribes audio from YouTube videos |
| [Send Email](email_block.md#send-email) | Sends emails using SMTP |
| [Condition Block](branching.md#condition-block) | Evaluates conditions for workflow branching |
| [Step Through Items](iteration.md#step-through-items) | Iterates through lists or dictionaries |

## Google Services
| Block Name | Description |
|------------|-------------|
| [Gmail Read](google/gmail.md#gmail-read) | Retrieves and reads emails from a Gmail account |
| [Gmail Send](google/gmail.md#gmail-send) | Sends emails using a Gmail account |
| [Gmail List Labels](google/gmail.md#gmail-list-labels) | Retrieves all labels from a Gmail account |
| [Gmail Add Label](google/gmail.md#gmail-add-label) | Adds a label to a specific email in a Gmail account |
| [Gmail Remove Label](google/gmail.md#gmail-remove-label) | Removes a label from a specific email in a Gmail account |
| [Google Sheets Read](google/sheet.md#google-sheets-read) | Reads data from a Google Sheets spreadsheet |
| [Google Sheets Write](google/sheet.md#google-sheets-write) | Writes data to a Google Sheets spreadsheet |
| [Google Maps Search](google_maps.md#google-maps-search) | Searches for local businesses using the Google Maps API |

## GitHub Integration
| Block Name | Description |
|------------|-------------|
| [GitHub Comment](github/issues.md#github-comment) | Posts comments on GitHub issues or pull requests |
| [GitHub Make Issue](github/issues.md#github-make-issue) | Creates new issues on GitHub repositories |
| [GitHub Read Issue](github/issues.md#github-read-issue) | Retrieves information about a specific GitHub issue |
| [GitHub List Issues](github/issues.md#github-list-issues) | Retrieves a list of issues from a GitHub repository |
| [GitHub Add Label](github/issues.md#github-add-label) | Adds a label to a GitHub issue or pull request |
| [GitHub Remove Label](github/issues.md#github-remove-label) | Removes a label from a GitHub issue or pull request |
| [GitHub Assign Issue](github/issues.md#github-assign-issue) | Assigns a user to a GitHub issue |
| [GitHub List Tags](github/repo.md#github-list-tags) | Retrieves and lists all tags for a specified GitHub repository |
| [GitHub List Branches](github/repo.md#github-list-branches) | Retrieves and lists all branches for a specified GitHub repository |
| [GitHub List Discussions](github/repo.md#github-list-discussions) | Retrieves and lists recent discussions for a specified GitHub repository |
| [GitHub Make Branch](github/repo.md#github-make-branch) | Creates a new branch in a GitHub repository |
| [GitHub Delete Branch](github/repo.md#github-delete-branch) | Deletes a specified branch from a GitHub repository |
| [GitHub List Pull Requests](github/pull_requests.md#github-list-pull-requests) | Retrieves a list of pull requests from a specified GitHub repository |
| [GitHub Make Pull Request](github/pull_requests.md#github-make-pull-request) | Creates a new pull request in a specified GitHub repository |
| [GitHub Read Pull Request](github/pull_requests.md#github-read-pull-request) | Retrieves detailed information about a specific GitHub pull request |
| [GitHub Assign PR Reviewer](github/pull_requests.md#github-assign-pr-reviewer) | Assigns a reviewer to a specific GitHub pull request |
| [GitHub Unassign PR Reviewer](github/pull_requests.md#github-unassign-pr-reviewer) | Removes an assigned reviewer from a specific GitHub pull request |
| [GitHub List PR Reviewers](github/pull_requests.md#github-list-pr-reviewers) | Retrieves a list of all assigned reviewers for a specific GitHub pull request |

This comprehensive list covers all the blocks available in AutoGPT. Each block is designed to perform a specific task, and they can be combined to create powerful, automated workflows. For more detailed information on each block, click on its name to view the full documentation.