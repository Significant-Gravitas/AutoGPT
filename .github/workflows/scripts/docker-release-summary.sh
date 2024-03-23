#!/bin/bash
meta=$(docker image inspect "$IMAGE_NAME" | jq '.[0]')

EOF=$(dd if=/dev/urandom bs=15 count=1 status=none | base64)

cat << $EOF
# Docker Release Build summary 🚀🔨

**Source:** $ref_type \`$current_ref\` -> [$repository@\`${commit_hash:0:7}\`]($source_url)

**Image size:** $((`jq -r .Size <<< $meta` / 10**6))MB

## Image details

**Tags:**
$(jq -r '.RepoTags | map("* `\(.)`") | join("\n")' <<< $meta)

<details>
<summary><h3>Layers</h3></summary>

|    Age    |  Size  | Created by instruction |
| --------- | ------ | ---------------------- |
$(docker history --no-trunc --format "{{.CreatedSince}}\t{{.Size}}\t\`{{.CreatedBy}}\`\t{{.Comment}}" $IMAGE_NAME \
    | grep 'buildkit.dockerfile' `# filter for layers created in this build process`\
    | cut -f-3                   `# yeet Comment column`\
    | sed 's/ ago//'             `# fix Layer age`\
    | sed 's/ # buildkit//'      `# remove buildkit comment from instructions`\
    | sed 's/\$/\\$/g'           `# escape variable and shell expansions`\
    | sed 's/|/\\|/g'            `# escape pipes so they don't interfere with column separators`\
    | column -t -s$'\t' -o' | '  `# align columns and add separator`\
    | sed 's/^/| /; s/$/ |/'     `# add table row start and end pipes`)
</details>

<details>
<summary><h3>ENV</h3></summary>

| Variable | Value    |
| -------- | -------- |
$(jq -r \
    '.Config.Env
    | map(
    split("=")
    | "\(.[0]) | `\(.[1] | gsub("\\s+"; " "))`"
    )
    | map("| \(.) |")
    | .[]' <<< $meta
)
</details>

<details>
<summary>Raw metadata</summary>

\`\`\`JSON
$meta
\`\`\`
</details>

## Build details
**Build trigger:** $event_name \`$current_ref\`

| Parameter      | Value        |
| -------------- | ------------ |
| \`no_cache\`   | \`$inputs_no_cache\` |

<details>
<summary><code>github</code> context</summary>

\`\`\`JSON
$github_context_json
\`\`\`
</details>

### Job environment

#### \`vars\` context:
\`\`\`JSON
$vars_json
\`\`\`

#### \`env\` context:
\`\`\`JSON
$job_env_json
\`\`\`

$EOF
