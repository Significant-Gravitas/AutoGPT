# Inventory

What the project actually uses, counted, before any decision is made.

## Count imports

For each old library, every component imported and how many files import
it:

```bash
grep -rhoE "from ['\"]@mui/material['\"]|import \{[^}]+\} from ['\"]@/components/ui/[a-z-]+['\"]" src \
  | sort | uniq -c | sort -rn
```

Adapt the pattern to the library. The output is the priority list: the ten
most-imported components are the primitives, and the long tail is what will
be decided case by case.

Then per screen, the set of components each route uses. A screen with three
components is a first candidate; a screen with twenty is last.

## The component map

Kobra's catalog, and what each common library calls the same thing. A dash
means the library has no equivalent and the screen gains something.

| Kobra           | shadcn/ui            | Material UI           | Chakra             | Ant Design            | Mantine          |
| --------------- | -------------------- | --------------------- | ------------------ | --------------------- | ---------------- |
| Accordion       | Accordion            | Accordion             | Accordion          | Collapse              | Accordion        |
| Alert           | Alert                | Alert                 | Alert              | Alert                 | Alert            |
| Aspect Ratio    | Aspect Ratio         | –                     | AspectRatio        | –                     | AspectRatio      |
| Avatar          | Avatar               | Avatar                | Avatar             | Avatar                | Avatar           |
| Badge           | Badge                | Chip                  | Badge, Tag         | Tag                   | Badge            |
| Breadcrumb      | Breadcrumb           | Breadcrumbs           | Breadcrumb         | Breadcrumb            | Breadcrumbs      |
| Button          | Button               | Button, IconButton    | Button, IconButton | Button                | Button           |
| Calendar        | Calendar             | DateCalendar (X)      | –                  | Calendar              | Calendar (dates) |
| Card            | Card                 | Card                  | Card               | Card                  | Card             |
| Carousel        | Carousel             | –                     | –                  | Carousel              | Carousel         |
| Chart           | Chart                | Charts (X)            | –                  | –                     | Charts           |
| Checkbox        | Checkbox             | Checkbox              | Checkbox           | Checkbox              | Checkbox         |
| Collapsible     | Collapsible          | Collapse              | Collapsible        | Collapse              | Collapse         |
| Combobox        | Combobox             | Autocomplete          | –                  | AutoComplete          | Autocomplete     |
| Multi Select    | –                    | Autocomplete multiple | –                  | Select multiple       | MultiSelect      |
| Command Menu    | Command              | –                     | –                  | –                     | Spotlight        |
| Context Menu    | Context Menu         | Menu (contextmenu)    | –                  | Dropdown              | –                |
| Dialog          | Dialog, Alert Dialog | Dialog                | Modal, AlertDialog | Modal                 | Modal            |
| Drawer          | Drawer               | SwipeableDrawer       | Drawer             | Drawer                | Drawer           |
| Dropdown Menu   | Dropdown Menu        | Menu                  | Menu               | Dropdown              | Menu             |
| Empty           | Empty                | –                     | –                  | Empty                 | –                |
| Form            | Form                 | FormControl           | FormControl        | Form                  | form             |
| Hover Card      | Hover Card           | –                     | Popover            | Popover               | HoverCard        |
| Input           | Input, Label         | TextField             | Input, FormLabel   | Input                 | TextInput        |
| Input OTP       | Input OTP            | –                     | PinInput           | OTP                   | PinInput         |
| Item            | Item                 | ListItem              | –                  | List.Item             | –                |
| Kbd             | Kbd                  | –                     | Kbd                | –                     | Kbd              |
| Menubar         | Menubar              | –                     | –                  | Menu horizontal       | –                |
| Navigation Menu | Navigation Menu      | –                     | –                  | Menu                  | NavLink          |
| Pagination      | Pagination           | Pagination            | –                  | Pagination            | Pagination       |
| Popover         | Popover              | Popover               | Popover            | Popover               | Popover          |
| Progress        | Progress             | LinearProgress        | Progress           | Progress              | Progress         |
| Radio Group     | Radio Group          | RadioGroup            | RadioGroup         | Radio.Group           | Radio.Group      |
| Resizable       | Resizable            | –                     | –                  | –                     | –                |
| Scroll Area     | Scroll Area          | –                     | –                  | –                     | ScrollArea       |
| Select          | Select               | Select                | Select             | Select                | Select           |
| Separator       | Separator            | Divider               | Divider            | Divider               | Divider          |
| Sheet           | Sheet                | Drawer                | Drawer             | Drawer                | Drawer           |
| Sidebar         | Sidebar              | Drawer permanent      | –                  | Layout.Sider          | AppShell.Navbar  |
| Skeleton        | Skeleton             | Skeleton              | Skeleton           | Skeleton              | Skeleton         |
| Slider          | Slider               | Slider                | Slider             | Slider                | Slider           |
| Sound           | –                    | –                     | –                  | –                     | –                |
| Spinner         | Spinner              | CircularProgress      | Spinner            | Spin                  | Loader           |
| Switch          | Switch               | Switch                | Switch             | Switch                | Switch           |
| Table           | Table                | Table                 | Table              | Table                 | Table            |
| Tabs            | Tabs                 | Tabs                  | Tabs               | Tabs                  | Tabs             |
| Textarea        | Textarea             | TextField multiline   | Textarea           | Input.TextArea        | Textarea         |
| Toast           | Sonner               | Snackbar              | useToast           | message, notification | Notifications    |
| Toggle Group    | Toggle Group         | ToggleButtonGroup     | –                  | Radio.Button          | SegmentedControl |
| Tooltip         | Tooltip              | Tooltip               | Tooltip            | Tooltip               | Tooltip          |

The AI and application components (Conversation, Message, Chat Input,
Streaming Text, Reasoning Steps, File Diff, Code Block, Task List, Plan Card,
Question Card, Inline Citations, Attachment, Dropzone, Image Generation, AI
Editor) and the effects (Halftone Dots, Logo Carousel, Input Dissolve,
Marker, Video, Direction) have no counterpart in any of these libraries.
They are additions, not replacements, and they go in after the migration
rather than during it.

## What has no target

Some things in the old library will map to nothing: a stepper, a rating, a
tree view, a date range picker with presets. For each, decide once and write
it in the ledger:

- **Compose it** from Kobra primitives (a stepper is Tabs with disabled
  triggers and a Progress under them).
- **Keep it** from the old library behind the compat layer, listed as a
  known remainder with an owner.
- **Drop it** if the inventory shows it is used once and the screen can do
  without.

## Custom behavior to preserve

Grep the old components for anything beyond the library: an `onKeyDown`
that does something particular, a debounce, an analytics call, a focus
trick. List each with its file and the Kobra component that will need to
carry it. These are the migration's real risk; the visual work is the easy
half.

## The output

A table in `MIGRATION.md`:

| Old component          | Files | Kobra target | Decision | Notes                              |
| ---------------------- | ----- | ------------ | -------- | ---------------------------------- |
| `@mui/material` Button | 84    | Button       | migrate  | `LoadingButton` → `BusySpinner`    |
| `@mui/material` Rating | 1     | –            | drop     | Only on the retired feedback page  |
| `@mui/x-date-pickers`  | 3     | Calendar     | compose  | Range presets rebuilt as a Popover |
