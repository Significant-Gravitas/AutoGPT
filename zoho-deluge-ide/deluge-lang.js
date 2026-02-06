// Deluge Language Definition for Monaco Editor

function registerDelugeLanguage() {
    monaco.languages.register({ id: 'deluge' });

    monaco.languages.setMonarchTokensProvider('deluge', {
        tokenizer: {
            root: [
                [/\b(if|else|for|each|in|return|break|continue|void|info|true|false|null)\b/, 'keyword'],
                [/\b(string|int|decimal|map|list|date|datetime|boolean)\b/, 'type'],
                [/"([^"\\]|\\.)*"/, 'string'],
                [/\/\/.*/, 'comment'],
                [/\/\*/, 'comment', '@comment'],
                [/[{}()\[\]]/, '@brackets'],
                [/[<>!=+\-*\/%]=?/, 'operator'],
                [/\b\d+(\.\d+)?\b/, 'number'],
                [/[a-zA-Z_][a-zA-Z0-9_]*/, 'variable'],
            ],
            comment: [
                [/[^\/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'],
                [/\*\//, 'comment', '@pop'],
                [/[\/*]/, 'comment']
            ],
        }
    });

    monaco.languages.setLanguageConfiguration('deluge', {
        brackets: [
            ['{', '}'],
            ['[', ']'],
            ['(', ')']
        ],
        autoClosingPairs: [
            { open: '{', close: '}' },
            { open: '[', close: ']' },
            { open: '(', close: ')' },
            { open: '"', close: '"' }
        ],
        surroundingPairs: [
            { open: '{', close: '}' },
            { open: '[', close: ']' },
            { open: '(', close: ')' },
            { open: '"', close: '"' }
        ],
        comments: {
            lineComment: '//',
            blockComment: ['/*', '*/']
        }
    });

    // Basic & Zoho API Autocomplete
    monaco.languages.registerCompletionItemProvider('deluge', {
        provideCompletionItems: (model, position) => {
            const word = model.getWordUntilPosition(position);
            const range = {
                startLineNumber: position.lineNumber,
                endLineNumber: position.lineNumber,
                startColumn: word.startColumn,
                endColumn: word.endColumn
            };

            const suggestions = [
                // Keywords
                { label: 'if', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'if (${1:condition}) {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'else', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'else {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'for each', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'for each ${1:variable} in ${2:collection} {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'return', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'return $0;', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'info', kind: monaco.languages.CompletionItemKind.Function, insertText: 'info $0;', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho CRM
                { label: 'zoho.crm.getRecordById', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.crm.getRecordById("${1:Module}", ${2:ID});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.crm.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.crm.getRecords("${1:Module}");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.crm.createRecord', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.crm.createRecord("${1:Module}", ${2:Map});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.crm.updateRecord', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.crm.updateRecord("${1:Module}", ${2:ID}, ${3:Map});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.crm.searchRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.crm.searchRecords("${1:Module}", "(${2:Criteria})");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho Creator
                { label: 'zoho.creator.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.creator.getRecords("${1:Owner}", "${2:App}", "${3:View}", ${4:Criteria});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.creator.createRecord', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.creator.createRecord("${1:Owner}", "${2:App}", "${3:Form}", ${4:Map});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.creator.updateRecord', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.creator.updateRecord("${1:Owner}", "${2:App}", "${3:View}", ${4:ID}, ${5:Map});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho Books
                { label: 'zoho.books.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.books.getRecords("${1:Module}", "${2:OrgID}");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'zoho.books.createRecord', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.books.createRecord("${1:Module}", "${2:OrgID}", ${3:Map});', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho Inventory
                { label: 'zoho.inventory.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.inventory.getRecords("${1:Module}", "${2:OrgID}");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho Recruit
                { label: 'zoho.recruit.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.recruit.getRecords("${1:Module}");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho Bigin
                { label: 'zoho.bigin.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.bigin.getRecords("${1:Module}");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Zoho Analytics
                { label: 'zoho.analytics.getRecords', kind: monaco.languages.CompletionItemKind.Method, insertText: 'zoho.analytics.getRecords("${1:DBName}", "${2:TableName}");', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

                // Common Deluge Methods
                { label: 'toString', kind: monaco.languages.CompletionItemKind.Method, insertText: 'toString()', range },
                { label: 'toNumber', kind: monaco.languages.CompletionItemKind.Method, insertText: 'toNumber()', range },
                { label: 'toDate', kind: monaco.languages.CompletionItemKind.Method, insertText: 'toDate()', range },
                { label: 'toLong', kind: monaco.languages.CompletionItemKind.Method, insertText: 'toLong()', range },
                { label: 'toMap', kind: monaco.languages.CompletionItemKind.Method, insertText: 'toMap()', range },
                { label: 'toList', kind: monaco.languages.CompletionItemKind.Method, insertText: 'toList()', range },
                { label: 'put', kind: monaco.languages.CompletionItemKind.Method, insertText: 'put(${1:key}, ${2:value})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'get', kind: monaco.languages.CompletionItemKind.Method, insertText: 'get(${1:key})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'add', kind: monaco.languages.CompletionItemKind.Method, insertText: 'add(${1:value})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
                { label: 'size', kind: monaco.languages.CompletionItemKind.Method, insertText: 'size()', range },
                { label: 'isEmpty', kind: monaco.languages.CompletionItemKind.Method, insertText: 'isEmpty()', range },
                { label: 'now', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'zoho.currentdate', range },
                { label: 'today', kind: monaco.languages.CompletionItemKind.Variable, insertText: 'zoho.currenttime', range },
            ];
            return { suggestions: suggestions };
        }
    });
}
