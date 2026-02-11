let sessionId = null;
let extractedData = [];
let mappings = [];
let selectedColumn = null;
let lastFile = null;

const api = {
    upload: async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        if (sessionId) formData.append('session_id', sessionId);

        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        return res.json();
    },
    train: async (mapping) => {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId, mapping })
        });
        return res.json();
    },
    finish: async () => {
        const res = await fetch('/api/finish_training', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId })
        });
        return res.json();
    },
    execute: async () => {
         if (!lastFile) return { status: 'error', message: 'No file' };

         const reader = new FileReader();
         return new Promise((resolve, reject) => {
             reader.onload = async () => {
                 const content_b64 = reader.result.split(',')[1];
                 const files = [{
                     filename: lastFile.name,
                     content_b64: content_b64
                 }];

                 const res = await fetch('/api/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ session_id: sessionId, files })
                });
                resolve(await res.json());
             };
             reader.readAsDataURL(lastFile);
         });
    },
    opportunities: async () => {
        const res = await fetch('/api/opportunities');
        return res.json();
    }
};

function log(msg, type='info') {
    const div = document.createElement('div');
    div.className = `log-entry log-${type}`;
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    const container = document.getElementById('logs-content');
    container.prepend(div);
}

function renderTable(data, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!data || data.length === 0) return;

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');

    // Headers
    const headers = Object.keys(data[0]);
    const trHead = document.createElement('tr');
    headers.forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);

    // Rows
    data.slice(0, 10).forEach((row, idx) => { // Limit to 10 for preview
        const tr = document.createElement('tr');
        headers.forEach(h => {
            const td = document.createElement('td');
            td.textContent = row[h];
            td.dataset.column = h;
            td.onclick = (e) => handleCellClick(e, h, row[h]);
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    container.appendChild(table);
}

function handleCellClick(e, column, value) {
    document.querySelectorAll('td.selected-cell').forEach(td => td.classList.remove('selected-cell'));
    e.target.classList.add('selected-cell');

    selectedColumn = column;
    document.getElementById('source-data-view').innerHTML = `
        <p><strong>Column:</strong> ${column}</p>
        <p><strong>Sample Value:</strong> ${value}</p>
    `;
}

document.getElementById('btn-upload').addEventListener('click', async () => {
    const fileInput = document.getElementById('file-upload');
    if (!fileInput.files.length) {
        log('Please select a file.', 'error');
        return;
    }

    lastFile = fileInput.files[0];
    log(`Uploading ${lastFile.name}...`);
    document.getElementById('status-text').textContent = "Uploading...";

    try {
        const data = await api.upload(lastFile);
        if (data.status === 'ok') {
            sessionId = data.session_id;
            extractedData = data.extracted;
            log(`Extracted ${extractedData.length} rows. Session: ${sessionId}`);
            renderTable(extractedData, 'data-table-container');
            document.getElementById('workspace').classList.remove('hidden');
            document.getElementById('status-text').textContent = "Data Extracted";
        } else {
            log(`Error: ${JSON.stringify(data)}`, 'error');
            document.getElementById('status-text').textContent = "Error";
        }
    } catch (e) {
        log(`Upload failed: ${e.message}`, 'error');
        document.getElementById('status-text').textContent = "Error";
    }
});

document.getElementById('btn-start-training').addEventListener('click', () => {
    switchTab('training');
    let tableContainer = document.getElementById('training-table-container');
    if (!tableContainer) {
        tableContainer = document.createElement('div');
        tableContainer.id = 'training-table-container';
        const tabContent = document.getElementById('tab-training');
        tabContent.insertBefore(tableContainer, tabContent.firstChild);
    }
    renderTable(extractedData, 'training-table-container');
});

document.getElementById('btn-add-mapping').addEventListener('click', () => {
    if (!selectedColumn) {
        log('Please select a source column (cell) first.', 'error');
        return;
    }
    const targetApp = document.getElementById('target-app').value;
    const targetField = document.getElementById('target-field').value;

    if (!targetField) {
        log('Please enter a target field.', 'error');
        return;
    }

    const mapping = {
        source: { file: lastFile.name, field: selectedColumn },
        target: { app: targetApp, field: targetField }
    };

    mappings.push(mapping);
    renderMappings();
    log(`Added mapping: ${selectedColumn} -> ${targetApp}.${targetField}`);

    selectedColumn = null;
    document.querySelectorAll('td.selected-cell').forEach(td => td.classList.remove('selected-cell'));
    document.getElementById('source-data-view').innerHTML = 'Click a cell to map';
    document.getElementById('target-field').value = '';
});

function renderMappings() {
    const list = document.getElementById('mapping-list');
    list.innerHTML = '';
    mappings.forEach((m, i) => {
        const li = document.createElement('li');
        li.textContent = `${m.source.field} -> ${m.target.app}.${m.target.field}`;
        list.appendChild(li);
    });
}

document.getElementById('btn-train-submit').addEventListener('click', async () => {
    if (mappings.length === 0) {
        log('No mappings to train.', 'error');
        return;
    }
    log('Submitting training...');
    document.getElementById('status-text').textContent = "Training...";
    try {
        const res = await api.train(mappings);
        if (res.status === 'ok') {
            log(`Training summary: ${res.summary}`);
            document.getElementById('btn-finish-training').disabled = false;
            document.getElementById('status-text').textContent = "Trained";
        } else {
            log(`Training error: ${JSON.stringify(res)}`, 'error');
        }
    } catch (e) {
        log(`Training failed: ${e.message}`, 'error');
    }
});

document.getElementById('btn-finish-training').addEventListener('click', async () => {
    log('Finishing training...');
    try {
        const res = await api.finish();
        if (res.status === 'ok') {
            log('Training complete! Execution plan received.');
            switchTab('execution');
            document.getElementById('status-text').textContent = "Ready to Execute";
        } else {
            log(`Finish error: ${JSON.stringify(res)}`, 'error');
        }
    } catch (e) {
        log(`Finish failed: ${e.message}`, 'error');
    }
});

document.getElementById('btn-execute').addEventListener('click', async () => {
    log('Executing...');
    document.getElementById('status-text').textContent = "Executing...";
    try {
        const res = await api.execute();
        if (res.status === 'ok') {
            log('Execution done!');
            document.getElementById('status-text').textContent = "Done";
            const resultsDiv = document.getElementById('execution-results');
            resultsDiv.innerHTML = '<pre>' + JSON.stringify(res.results, null, 2) + '</pre>';
        } else {
            log(`Execution error: ${JSON.stringify(res)}`, 'error');
        }
    } catch (e) {
        log(`Execution failed: ${e.message}`, 'error');
    }
});

document.getElementById('btn-fetch-opportunities').addEventListener('click', async () => {
    log('Fetching opportunities...');
    try {
        const res = await api.opportunities();
        if (res.status === 'ok') {
            const list = document.getElementById('opportunities-list');
            list.innerHTML = '';
            res.opportunities.forEach(op => {
                const li = document.createElement('li');
                li.innerHTML = `<strong>${op.title}</strong><p>${op.summary}</p>`;
                list.appendChild(li);
            });
            document.getElementById('opportunities-section').classList.remove('hidden');
        }
    } catch (e) {
        log(`Opportunities failed: ${e.message}`, 'error');
    }
});

document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        switchTab(btn.dataset.tab);
    });
});

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
}
