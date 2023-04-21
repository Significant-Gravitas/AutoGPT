// Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
// For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

// Coverage.py HTML report browser code.
/*jslint browser: true, sloppy: true, vars: true, plusplus: true, maxerr: 50, indent: 4 */
/*global coverage: true, document, window, $ */

coverage = {};

// General helpers
function debounce(callback, wait) {
    let timeoutId = null;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            callback.apply(this, args);
        }, wait);
    };
};

function checkVisible(element) {
    const rect = element.getBoundingClientRect();
    const viewBottom = Math.max(document.documentElement.clientHeight, window.innerHeight);
    const viewTop = 30;
    return !(rect.bottom < viewTop || rect.top >= viewBottom);
}

function on_click(sel, fn) {
    const elt = document.querySelector(sel);
    if (elt) {
        elt.addEventListener("click", fn);
    }
}

// Helpers for table sorting
function getCellValue(row, column = 0) {
    const cell = row.cells[column]
    if (cell.childElementCount == 1) {
        const child = cell.firstElementChild
        if (child instanceof HTMLTimeElement && child.dateTime) {
            return child.dateTime
        } else if (child instanceof HTMLDataElement && child.value) {
            return child.value
        }
    }
    return cell.innerText || cell.textContent;
}

function rowComparator(rowA, rowB, column = 0) {
    let valueA = getCellValue(rowA, column);
    let valueB = getCellValue(rowB, column);
    if (!isNaN(valueA) && !isNaN(valueB)) {
        return valueA - valueB
    }
    return valueA.localeCompare(valueB, undefined, {numeric: true});
}

function sortColumn(th) {
    // Get the current sorting direction of the selected header,
    // clear state on other headers and then set the new sorting direction
    const currentSortOrder = th.getAttribute("aria-sort");
    [...th.parentElement.cells].forEach(header => header.setAttribute("aria-sort", "none"));
    if (currentSortOrder === "none") {
        th.setAttribute("aria-sort", th.dataset.defaultSortOrder || "ascending");
    } else {
        th.setAttribute("aria-sort", currentSortOrder === "ascending" ? "descending" : "ascending");
    }

    const column = [...th.parentElement.cells].indexOf(th)

    // Sort all rows and afterwards append them in order to move them in the DOM
    Array.from(th.closest("table").querySelectorAll("tbody tr"))
        .sort((rowA, rowB) => rowComparator(rowA, rowB, column) * (th.getAttribute("aria-sort") === "ascending" ? 1 : -1))
        .forEach(tr => tr.parentElement.appendChild(tr) );
}

// Find all the elements with data-shortcut attribute, and use them to assign a shortcut key.
coverage.assign_shortkeys = function () {
    document.querySelectorAll("[data-shortcut]").forEach(element => {
        document.addEventListener("keypress", event => {
            if (event.target.tagName.toLowerCase() === "input") {
                return; // ignore keypress from search filter
            }
            if (event.key === element.dataset.shortcut) {
                element.click();
            }
        });
    });
};

// Create the events for the filter box.
coverage.wire_up_filter = function () {
    // Cache elements.
    const table = document.querySelector("table.index");
    const table_body_rows = table.querySelectorAll("tbody tr");
    const no_rows = document.getElementById("no_rows");

    // Observe filter keyevents.
    document.getElementById("filter").addEventListener("input", debounce(event => {
        // Keep running total of each metric, first index contains number of shown rows
        const totals = new Array(table.rows[0].cells.length).fill(0);
        // Accumulate the percentage as fraction
        totals[totals.length - 1] = { "numer": 0, "denom": 0 };

        // Hide / show elements.
        table_body_rows.forEach(row => {
            if (!row.cells[0].textContent.includes(event.target.value)) {
                // hide
                row.classList.add("hidden");
                return;
            }

            // show
            row.classList.remove("hidden");
            totals[0]++;

            for (let column = 1; column < totals.length; column++) {
                // Accumulate dynamic totals
                cell = row.cells[column]
                if (column === totals.length - 1) {
                    // Last column contains percentage
                    const [numer, denom] = cell.dataset.ratio.split(" ");
                    totals[column]["numer"] += parseInt(numer, 10);
                    totals[column]["denom"] += parseInt(denom, 10);
                } else {
                    totals[column] += parseInt(cell.textContent, 10);
                }
            }
        });

        // Show placeholder if no rows will be displayed.
        if (!totals[0]) {
            // Show placeholder, hide table.
            no_rows.style.display = "block";
            table.style.display = "none";
            return;
        }

        // Hide placeholder, show table.
        no_rows.style.display = null;
        table.style.display = null;

        const footer = table.tFoot.rows[0];
        // Calculate new dynamic sum values based on visible rows.
        for (let column = 1; column < totals.length; column++) {
            // Get footer cell element.
            const cell = footer.cells[column];

            // Set value into dynamic footer cell element.
            if (column === totals.length - 1) {
                // Percentage column uses the numerator and denominator,
                // and adapts to the number of decimal places.
                const match = /\.([0-9]+)/.exec(cell.textContent);
                const places = match ? match[1].length : 0;
                const { numer, denom } = totals[column];
                cell.dataset.ratio = `${numer} ${denom}`;
                // Check denom to prevent NaN if filtered files contain no statements
                cell.textContent = denom
                    ? `${(numer * 100 / denom).toFixed(places)}%`
                    : `${(100).toFixed(places)}%`;
            } else {
                cell.textContent = totals[column];
            }
        }
    }));

    // Trigger change event on setup, to force filter on page refresh
    // (filter value may still be present).
    document.getElementById("filter").dispatchEvent(new Event("input"));
};

coverage.INDEX_SORT_STORAGE = "COVERAGE_INDEX_SORT_2";

// Loaded on index.html
coverage.index_ready = function () {
    coverage.assign_shortkeys();
    coverage.wire_up_filter();
    document.querySelectorAll("[data-sortable] th[aria-sort]").forEach(
        th => th.addEventListener("click", e => sortColumn(e.target))
    );

    // Look for a localStorage item containing previous sort settings:
    const stored_list = localStorage.getItem(coverage.INDEX_SORT_STORAGE);

    if (stored_list) {
        const {column, direction} = JSON.parse(stored_list);
        const th = document.querySelector("[data-sortable]").tHead.rows[0].cells[column];
        th.setAttribute("aria-sort", direction === "ascending" ? "descending" : "ascending");
        th.click()
    }

    // Watch for page unload events so we can save the final sort settings:
    window.addEventListener("unload", function () {
        const th = document.querySelector('[data-sortable] th[aria-sort="ascending"], [data-sortable] [aria-sort="descending"]');
        if (!th) {
            return;
        }
        localStorage.setItem(coverage.INDEX_SORT_STORAGE, JSON.stringify({
            column: [...th.parentElement.cells].indexOf(th),
            direction: th.getAttribute("aria-sort"),
        }));
    });

    on_click(".button_prev_file", coverage.to_prev_file);
    on_click(".button_next_file", coverage.to_next_file);

    on_click(".button_show_hide_help", coverage.show_hide_help);
};

// -- pyfile stuff --

coverage.LINE_FILTERS_STORAGE = "COVERAGE_LINE_FILTERS";

coverage.pyfile_ready = function () {
    // If we're directed to a particular line number, highlight the line.
    var frag = location.hash;
    if (frag.length > 2 && frag[1] === "t") {
        document.querySelector(frag).closest(".n").classList.add("highlight");
        coverage.set_sel(parseInt(frag.substr(2), 10));
    } else {
        coverage.set_sel(0);
    }

    on_click(".button_toggle_run", coverage.toggle_lines);
    on_click(".button_toggle_mis", coverage.toggle_lines);
    on_click(".button_toggle_exc", coverage.toggle_lines);
    on_click(".button_toggle_par", coverage.toggle_lines);

    on_click(".button_next_chunk", coverage.to_next_chunk_nicely);
    on_click(".button_prev_chunk", coverage.to_prev_chunk_nicely);
    on_click(".button_top_of_page", coverage.to_top);
    on_click(".button_first_chunk", coverage.to_first_chunk);

    on_click(".button_prev_file", coverage.to_prev_file);
    on_click(".button_next_file", coverage.to_next_file);
    on_click(".button_to_index", coverage.to_index);

    on_click(".button_show_hide_help", coverage.show_hide_help);

    coverage.filters = undefined;
    try {
        coverage.filters = localStorage.getItem(coverage.LINE_FILTERS_STORAGE);
    } catch(err) {}

    if (coverage.filters) {
        coverage.filters = JSON.parse(coverage.filters);
    }
    else {
        coverage.filters = {run: false, exc: true, mis: true, par: true};
    }

    for (cls in coverage.filters) {
        coverage.set_line_visibilty(cls, coverage.filters[cls]);
    }

    coverage.assign_shortkeys();
    coverage.init_scroll_markers();
    coverage.wire_up_sticky_header();

    document.querySelectorAll("[id^=ctxs]").forEach(
        cbox => cbox.addEventListener("click", coverage.expand_contexts)
    );

    // Rebuild scroll markers when the window height changes.
    window.addEventListener("resize", coverage.build_scroll_markers);
};

coverage.toggle_lines = function (event) {
    const btn = event.target.closest("button");
    const category = btn.value
    const show = !btn.classList.contains("show_" + category);
    coverage.set_line_visibilty(category, show);
    coverage.build_scroll_markers();
    coverage.filters[category] = show;
    try {
        localStorage.setItem(coverage.LINE_FILTERS_STORAGE, JSON.stringify(coverage.filters));
    } catch(err) {}
};

coverage.set_line_visibilty = function (category, should_show) {
    const cls = "show_" + category;
    const btn = document.querySelector(".button_toggle_" + category);
    if (btn) {
        if (should_show) {
            document.querySelectorAll("#source ." + category).forEach(e => e.classList.add(cls));
            btn.classList.add(cls);
        }
        else {
            document.querySelectorAll("#source ." + category).forEach(e => e.classList.remove(cls));
            btn.classList.remove(cls);
        }
    }
};

// Return the nth line div.
coverage.line_elt = function (n) {
    return document.getElementById("t" + n)?.closest("p");
};

// Set the selection.  b and e are line numbers.
coverage.set_sel = function (b, e) {
    // The first line selected.
    coverage.sel_begin = b;
    // The next line not selected.
    coverage.sel_end = (e === undefined) ? b+1 : e;
};

coverage.to_top = function () {
    coverage.set_sel(0, 1);
    coverage.scroll_window(0);
};

coverage.to_first_chunk = function () {
    coverage.set_sel(0, 1);
    coverage.to_next_chunk();
};

coverage.to_prev_file = function () {
    window.location = document.getElementById("prevFileLink").href;
}

coverage.to_next_file = function () {
    window.location = document.getElementById("nextFileLink").href;
}

coverage.to_index = function () {
    location.href = document.getElementById("indexLink").href;
}

coverage.show_hide_help = function () {
    const helpCheck = document.getElementById("help_panel_state")
    helpCheck.checked = !helpCheck.checked;
}

// Return a string indicating what kind of chunk this line belongs to,
// or null if not a chunk.
coverage.chunk_indicator = function (line_elt) {
    const classes = line_elt?.className;
    if (!classes) {
        return null;
    }
    const match = classes.match(/\bshow_\w+\b/);
    if (!match) {
        return null;
    }
    return match[0];
};

coverage.to_next_chunk = function () {
    const c = coverage;

    // Find the start of the next colored chunk.
    var probe = c.sel_end;
    var chunk_indicator, probe_line;
    while (true) {
        probe_line = c.line_elt(probe);
        if (!probe_line) {
            return;
        }
        chunk_indicator = c.chunk_indicator(probe_line);
        if (chunk_indicator) {
            break;
        }
        probe++;
    }

    // There's a next chunk, `probe` points to it.
    var begin = probe;

    // Find the end of this chunk.
    var next_indicator = chunk_indicator;
    while (next_indicator === chunk_indicator) {
        probe++;
        probe_line = c.line_elt(probe);
        next_indicator = c.chunk_indicator(probe_line);
    }
    c.set_sel(begin, probe);
    c.show_selection();
};

coverage.to_prev_chunk = function () {
    const c = coverage;

    // Find the end of the prev colored chunk.
    var probe = c.sel_begin-1;
    var probe_line = c.line_elt(probe);
    if (!probe_line) {
        return;
    }
    var chunk_indicator = c.chunk_indicator(probe_line);
    while (probe > 1 && !chunk_indicator) {
        probe--;
        probe_line = c.line_elt(probe);
        if (!probe_line) {
            return;
        }
        chunk_indicator = c.chunk_indicator(probe_line);
    }

    // There's a prev chunk, `probe` points to its last line.
    var end = probe+1;

    // Find the beginning of this chunk.
    var prev_indicator = chunk_indicator;
    while (prev_indicator === chunk_indicator) {
        probe--;
        if (probe <= 0) {
            return;
        }
        probe_line = c.line_elt(probe);
        prev_indicator = c.chunk_indicator(probe_line);
    }
    c.set_sel(probe+1, end);
    c.show_selection();
};

// Returns 0, 1, or 2: how many of the two ends of the selection are on
// the screen right now?
coverage.selection_ends_on_screen = function () {
    if (coverage.sel_begin === 0) {
        return 0;
    }

    const begin = coverage.line_elt(coverage.sel_begin);
    const end = coverage.line_elt(coverage.sel_end-1);

    return (
        (checkVisible(begin) ? 1 : 0)
        + (checkVisible(end) ? 1 : 0)
    );
};

coverage.to_next_chunk_nicely = function () {
    if (coverage.selection_ends_on_screen() === 0) {
        // The selection is entirely off the screen:
        // Set the top line on the screen as selection.

        // This will select the top-left of the viewport
        // As this is most likely the span with the line number we take the parent
        const line = document.elementFromPoint(0, 0).parentElement;
        if (line.parentElement !== document.getElementById("source")) {
            // The element is not a source line but the header or similar
            coverage.select_line_or_chunk(1);
        } else {
            // We extract the line number from the id
            coverage.select_line_or_chunk(parseInt(line.id.substring(1), 10));
        }
    }
    coverage.to_next_chunk();
};

coverage.to_prev_chunk_nicely = function () {
    if (coverage.selection_ends_on_screen() === 0) {
        // The selection is entirely off the screen:
        // Set the lowest line on the screen as selection.

        // This will select the bottom-left of the viewport
        // As this is most likely the span with the line number we take the parent
        const line = document.elementFromPoint(document.documentElement.clientHeight-1, 0).parentElement;
        if (line.parentElement !== document.getElementById("source")) {
            // The element is not a source line but the header or similar
            coverage.select_line_or_chunk(coverage.lines_len);
        } else {
            // We extract the line number from the id
            coverage.select_line_or_chunk(parseInt(line.id.substring(1), 10));
        }
    }
    coverage.to_prev_chunk();
};

// Select line number lineno, or if it is in a colored chunk, select the
// entire chunk
coverage.select_line_or_chunk = function (lineno) {
    var c = coverage;
    var probe_line = c.line_elt(lineno);
    if (!probe_line) {
        return;
    }
    var the_indicator = c.chunk_indicator(probe_line);
    if (the_indicator) {
        // The line is in a highlighted chunk.
        // Search backward for the first line.
        var probe = lineno;
        var indicator = the_indicator;
        while (probe > 0 && indicator === the_indicator) {
            probe--;
            probe_line = c.line_elt(probe);
            if (!probe_line) {
                break;
            }
            indicator = c.chunk_indicator(probe_line);
        }
        var begin = probe + 1;

        // Search forward for the last line.
        probe = lineno;
        indicator = the_indicator;
        while (indicator === the_indicator) {
            probe++;
            probe_line = c.line_elt(probe);
            indicator = c.chunk_indicator(probe_line);
        }

        coverage.set_sel(begin, probe);
    }
    else {
        coverage.set_sel(lineno);
    }
};

coverage.show_selection = function () {
    // Highlight the lines in the chunk
    document.querySelectorAll("#source .highlight").forEach(e => e.classList.remove("highlight"));
    for (let probe = coverage.sel_begin; probe < coverage.sel_end; probe++) {
        coverage.line_elt(probe).querySelector(".n").classList.add("highlight");
    }

    coverage.scroll_to_selection();
};

coverage.scroll_to_selection = function () {
    // Scroll the page if the chunk isn't fully visible.
    if (coverage.selection_ends_on_screen() < 2) {
        const element = coverage.line_elt(coverage.sel_begin);
        coverage.scroll_window(element.offsetTop - 60);
    }
};

coverage.scroll_window = function (to_pos) {
    window.scroll({top: to_pos, behavior: "smooth"});
};

coverage.init_scroll_markers = function () {
    // Init some variables
    coverage.lines_len = document.querySelectorAll("#source > p").length;

    // Build html
    coverage.build_scroll_markers();
};

coverage.build_scroll_markers = function () {
    const temp_scroll_marker = document.getElementById("scroll_marker")
    if (temp_scroll_marker) temp_scroll_marker.remove();
    // Don't build markers if the window has no scroll bar.
    if (document.body.scrollHeight <= window.innerHeight) {
        return;
    }

    const marker_scale = window.innerHeight / document.body.scrollHeight;
    const line_height = Math.min(Math.max(3, window.innerHeight / coverage.lines_len), 10);

    let previous_line = -99, last_mark, last_top;

    const scroll_marker = document.createElement("div");
    scroll_marker.id = "scroll_marker";
    document.getElementById("source").querySelectorAll(
        "p.show_run, p.show_mis, p.show_exc, p.show_exc, p.show_par"
    ).forEach(element => {
        const line_top = Math.floor(element.offsetTop * marker_scale);
        const line_number = parseInt(element.querySelector(".n a").id.substr(1));

        if (line_number === previous_line + 1) {
            // If this solid missed block just make previous mark higher.
            last_mark.style.height = `${line_top + line_height - last_top}px`;
        } else {
            // Add colored line in scroll_marker block.
            last_mark = document.createElement("div");
            last_mark.id = `m${line_number}`;
            last_mark.classList.add("marker");
            last_mark.style.height = `${line_height}px`;
            last_mark.style.top = `${line_top}px`;
            scroll_marker.append(last_mark);
            last_top = line_top;
        }

        previous_line = line_number;
    });

    // Append last to prevent layout calculation
    document.body.append(scroll_marker);
};

coverage.wire_up_sticky_header = function () {
    const header = document.querySelector("header");
    const header_bottom = (
        header.querySelector(".content h2").getBoundingClientRect().top -
        header.getBoundingClientRect().top
    );

    function updateHeader() {
        if (window.scrollY > header_bottom) {
            header.classList.add("sticky");
        } else {
            header.classList.remove("sticky");
        }
    }

    window.addEventListener("scroll", updateHeader);
    updateHeader();
};

coverage.expand_contexts = function (e) {
    var ctxs = e.target.parentNode.querySelector(".ctxs");

    if (!ctxs.classList.contains("expanded")) {
        var ctxs_text = ctxs.textContent;
        var width = Number(ctxs_text[0]);
        ctxs.textContent = "";
        for (var i = 1; i < ctxs_text.length; i += width) {
            key = ctxs_text.substring(i, i + width).trim();
            ctxs.appendChild(document.createTextNode(contexts[key]));
            ctxs.appendChild(document.createElement("br"));
        }
        ctxs.classList.add("expanded");
    }
};

document.addEventListener("DOMContentLoaded", () => {
    if (document.body.classList.contains("indexfile")) {
        coverage.index_ready();
    } else {
        coverage.pyfile_ready();
    }
});
