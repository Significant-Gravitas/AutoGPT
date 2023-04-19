import curses


def safe_addstr(stdscr, y, x, text, attr=0):
    max_y, max_x = stdscr.getmaxyx()
    if y < max_y and x < max_x:
        stdscr.addstr(y, x, text, attr)

def main(stdscr):
    # Configuration de base pour curses
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

    arguments = [
        '--ai-settings <filename>',
        '--use-memory <memory-backend>',
        '--speak',
        '--debug',
        '--gpt3only',
        '--continuous',
        '--file <FILE>',
        '--dir <DIR>',
        '--init',
        '--overlap <OVERLAP>',
        '--max_length <MAX_LENGTH>'
    ]
    langues = ['fr', 'en', 'de', 'es', 'it']
    index_langue = 0
    descriptions = {
        'en': [
            'Run Auto-GPT with a different AI settings file',
            'Specify a memory management system',
            'Use TTS (Text-to-Speech) for Auto-GPT',
            'Display debug logs',
            'Use only GPT-3 if you do not have access to GPT-4 API',
            'Run the AI without user permission, in 100% automated mode (not recommended and potentially dangerous)',
            'Ingest a single file into memory',
            'Ingest all files from a directory into memory',
            'Initialize memory and clear its contents',
            'Overlap size between pieces when ingesting files',
            'Maximum length of each piece when ingesting files'
        ],
        'fr': [
            'Exécutez Auto-GPT avec un fichier de paramètres AI différent',
            'Spécifiez un système de gestion de mémoire',
            'Utilisez TTS (Text-to-Speech) pour Auto-GPT',
            'Affichez les journaux de débogage',
            'Utilisez uniquement GPT-3 si vous n\'avez pas accès à l\'API GPT-4',
            'Exécutez l\'IA sans autorisation de l\'utilisateur, en mode 100% automatisé (non recommandé et potentiellement dangereux)',
            'Ingestion d\'un seul fichier dans la mémoire',
            'Ingestion de tous les fichiers d\'un répertoire dans la mémoire',
            'Initialise la mémoire et efface son contenu',
            'OVERLAP: La taille de chevauchement entre les morceaux lors de l\'ingestion de fichiers',
            'La longueur maximale de chaque morceau lors de l\'ingestion de fichiers'
        ],
        'de': [
            'Führen Sie Auto-GPT mit einer anderen KI-Einstellungsdatei aus',
            'Geben Sie ein Speichermanagementsystem an',
            'Verwenden Sie TTS (Text-to-Speech) für Auto-GPT',
            'Debug-Protokolle anzeigen',
            'Verwenden Sie nur GPT-3, wenn Sie keinen Zugriff auf die GPT-4-API haben',
            'Führen Sie die KI ohne Benutzerberechtigung im 100% automatisierten Modus aus (nicht empfohlen und möglicherweise gefährlich)',
            'Einzelne Datei in den Speicher einlesen',
            'Alle Dateien aus einem Verzeichnis in den Speicher einlesen',
            'Initialisieren des Speichers und Löschen seines Inhalts',
            'Überlappung: Überlappungsgröße zwischen den Teilen beim Einlesen von Dateien',
            'Maximale Länge jedes Teils beim Einlesen von Dateien'
        ],
        'es': [
            'Ejecutar Auto-GPT con un archivo de configuración de IA diferente',
            'Especificar un sistema de gestión de memoria',
            'Usar TTS (Texto a Voz) para Auto-GPT',
            'Mostrar registros de depuración',
            'Usar solo GPT-3 si no tiene acceso a la API de GPT-4',
            'Ejecutar la IA sin permiso del usuario, en modo 100% automatizado (no recomendado y potencialmente peligroso)',
            'Ingestar un único archivo en la memoria',
            'Ingestar todos los archivos de un directorio en la memoria',
            'Inicializar la memoria y borrar su contenido',
            'Solapamiento: tamaño de solapamiento entre piezas al ingerir archivos',
            'Longitud máxima de cada pieza al ingerir archivos'
        ],
        'it': [
            'Esegui Auto-GPT con un file di impostazioni IA diverso',
            'Specifica un sistema di gestione della memoria',
            'Usa TTS (Text-to-Speech) per Auto-GPT',
            'Visualizza i log di debug',
            'Usa solo GPT-3 se non si ha accesso all\'API GPT-4',
            'Esegui l\'IA senza permesso dell\'utente, in modalità 100% automatizzata (non consigliato e potenzialmente pericoloso)',
            'Ingestione di un singolo file nella memoria',
            'Ingestione di tutti i file di una directory nella memoria',
            'Inizializza la memoria e cancella il suo contenuto',
            'Sovrapposizione: dimensione della sovrapposizione tra i pezzi durante l\'ingestione di file',
            'Lunghezza massima di ogni pezzo durante l\'ingestione di file'
        ],
    }
    instructions = {
        'fr': [
            "↑/↓: Changer la sélection",
            "Espace: Valider/Invalider la sélection",
            "←/→: Changer la langue des descriptions",
            "Enter: lancer le programme."
        ],
        'en': [
            "↑/↓: Change selection",
            "Space: Toggle selection",
            "←/→: Change description language",
            "Enter: launch the program."
        ],
        'de': [
            "↑/↓: Auswahl ändern",
            "Leertaste: Auswahl umschalten",
            "←/→: Beschreibungssprache ändern",
            "Enter: Programm starten."
        ],
        'es': [
            "↑/↓: Cambiar selección",
            "Espacio: Activar/desactivar selección",
            "←/→: Cambiar idioma de las descripciones",
            "Enter: iniciar el programa."
        ],
        'it': [
            "↑/↓: Cambia selezione",
            "Spazio: Attiva/disattiva selezione",
            "←/→: Cambia lingua delle descrizioni",
            "Enter: avvia il programma."
        ]
    }
    
    index = 0
    selected = [False] * len(arguments)
    language = 'fr'
    cmdline = ""
    cmdText = {
        'fr': [
            "Ligne de commande: "
        ],
        'en': [
            "Command line: "
        ],
        'de': [
            "Befehlszeile: "
        ],
        'es': [
            "Línea de comandos: "
        ],
        'it': [
            "Riga di comando: "
        ]
    }

    while True:
        stdscr.clear()

        # Afficher l'en-tête
        safe_addstr(stdscr, 0, 0, "Auto-GPT Launcher", curses.A_BOLD)

        # Afficher les instructions d'utilisation
        for i, instruction in enumerate(instructions[language]):
            safe_addstr(stdscr, 1 + i, 0, instruction)

    # Afficher les arguments et mettre en surbrillance l'argument sélectionné
        for i, argument in enumerate(arguments):
            if i == index:
                stdscr.attron(curses.A_REVERSE)
            if selected[i]:
                stdscr.attron(curses.A_BOLD)
            stdscr.addstr(i +6, 0, argument)
            if i == index:
                stdscr.attroff(curses.A_REVERSE)
            if selected[i]:
                stdscr.attroff(curses.A_BOLD)
            stdscr.addstr(i +6, len(argument) + 2, descriptions[language][i])

        # Afficher la ligne de commande construite
        stdscr.addstr(len(arguments) + 7, 0, cmdText[language][0] + cmdline)

        # Récupérer la touche appuyée
        key = stdscr.getch()

        if key == curses.KEY_UP and index > 0:
            index -= 1
        elif key == curses.KEY_DOWN and index < len(arguments) - 1:
            index += 1
         # Gérer la touche espace pour sélectionner ou désélectionner un argument
        elif key == ord(' '):
            selected[index] = not selected[index]
            cmdline = " ".join(argument for i, argument in enumerate(arguments) if selected[i])
        elif key == curses.KEY_LEFT:
            index_langue = (index_langue - 1) % len(langues)
            language = langues[index_langue]
        elif key == curses.KEY_RIGHT:
            index_langue = (index_langue + 1) % len(langues)
            language = langues[index_langue]
        elif key == ord('q'):
            break
        elif key == curses.KEY_ENTER or key == 10 or key == 13:
            return cmdline


cmdline = curses.wrapper(main)
print(f"python3 -m autogpt {cmdline}")

