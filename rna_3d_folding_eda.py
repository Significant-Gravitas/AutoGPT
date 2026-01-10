# %% [markdown]
# # 🧬 Stanford RNA 3D Folding Part 2 - EDA & Visualisation 3D
#
# Ce notebook explore les données du concours Stanford RNA 3D Folding Part 2.
#
# **Objectif** : Prédire la structure 3D de molécules d'ARN à partir de leur séquence.
#
# **Sections** :
# 1. Chargement des données
# 2. Analyse exploratoire des séquences
# 3. Analyse des labels (coordonnées 3D)
# 4. Visualisation 3D des structures RNA
# 5. Analyse des MSA (Multiple Sequence Alignments)
# 6. Exploration des métadonnées

# %% [markdown]
# ## 📦 1. Imports et Configuration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================
# Fonction simple pour parser les fichiers FASTA (sans BioPython)
# ============================================
def parse_fasta(fasta_path):
    """Parse un fichier FASTA et retourne une liste de tuples (header, sequence)."""
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]  # Enlever le '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Ajouter la dernière séquence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))

    return sequences

# Configuration des couleurs pour les nucléotides
NUCLEOTIDE_COLORS = {
    'A': '#FF6B6B',  # Rouge - Adénine
    'U': '#4ECDC4',  # Cyan - Uracile
    'G': '#45B7D1',  # Bleu - Guanine
    'C': '#96CEB4',  # Vert - Cytosine
}

# Chemin vers les données (ajuster selon l'environnement Kaggle)
DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')

print("📁 Structure des données:")
print(f"  - MSA/: {len(list((DATA_PATH / 'MSA').glob('*.fasta')))} fichiers")
print(f"  - PDB_RNA/: {len(list((DATA_PATH / 'PDB_RNA').glob('*.cif')))} fichiers")

# %% [markdown]
# ## 📊 2. Chargement des Données

# %%
# Chargement des fichiers CSV principaux
train_sequences = pd.read_csv(DATA_PATH / 'train_sequences.csv')
validation_sequences = pd.read_csv(DATA_PATH / 'validation_sequences.csv')
test_sequences = pd.read_csv(DATA_PATH / 'test_sequences.csv')

train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')
validation_labels = pd.read_csv(DATA_PATH / 'validation_labels.csv')

sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv')

# Métadonnées
rna_metadata = pd.read_csv(DATA_PATH / 'extra' / 'rna_metadata.csv')

print("=" * 60)
print("📋 RÉSUMÉ DES DONNÉES")
print("=" * 60)
print(f"\n🔹 Séquences:")
print(f"   Train:       {len(train_sequences):,} séquences")
print(f"   Validation:  {len(validation_sequences):,} séquences")
print(f"   Test:        {len(test_sequences):,} séquences")
print(f"\n🔹 Labels (coordonnées 3D):")
print(f"   Train:       {len(train_labels):,} résidus")
print(f"   Validation:  {len(validation_labels):,} résidus")
print(f"\n🔹 Métadonnées: {len(rna_metadata):,} entrées")

# %% [markdown]
# ## 🔬 3. Analyse des Séquences

# %% [markdown]
# ### 3.1 Structure des données de séquences

# %%
print("📋 Colonnes train_sequences:")
print(train_sequences.columns.tolist())
print("\n" + "=" * 60)
train_sequences.head(3)

# %%
# Aperçu détaillé d'une séquence
print("🔍 Exemple de séquence (première entrée):")
print("-" * 60)
for col in train_sequences.columns:
    val = train_sequences.iloc[0][col]
    if isinstance(val, str) and len(val) > 100:
        print(f"{col}: {val[:100]}...")
    else:
        print(f"{col}: {val}")

# %% [markdown]
# ### 3.2 Distribution des longueurs de séquences

# %%
# Calculer les longueurs
train_sequences['seq_length'] = train_sequences['sequence'].str.len()
validation_sequences['seq_length'] = validation_sequences['sequence'].str.len()
test_sequences['seq_length'] = test_sequences['sequence'].str.len()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Train
axes[0].hist(train_sequences['seq_length'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title(f'Train (n={len(train_sequences):,})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Longueur de séquence')
axes[0].set_ylabel('Fréquence')
axes[0].axvline(train_sequences['seq_length'].median(), color='red', linestyle='--', label=f"Médiane: {train_sequences['seq_length'].median():.0f}")
axes[0].legend()

# Validation
axes[1].hist(validation_sequences['seq_length'], bins=30, color='seagreen', alpha=0.7, edgecolor='black')
axes[1].set_title(f'Validation (n={len(validation_sequences):,})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Longueur de séquence')
axes[1].axvline(validation_sequences['seq_length'].median(), color='red', linestyle='--', label=f"Médiane: {validation_sequences['seq_length'].median():.0f}")
axes[1].legend()

# Test
axes[2].hist(test_sequences['seq_length'], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[2].set_title(f'Test (n={len(test_sequences):,})', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Longueur de séquence')
axes[2].axvline(test_sequences['seq_length'].median(), color='red', linestyle='--', label=f"Médiane: {test_sequences['seq_length'].median():.0f}")
axes[2].legend()

plt.suptitle('📊 Distribution des Longueurs de Séquences RNA', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Statistiques
print("\n📈 Statistiques des longueurs:")
print("-" * 60)
for name, df in [('Train', train_sequences), ('Validation', validation_sequences), ('Test', test_sequences)]:
    print(f"{name:12} | Min: {df['seq_length'].min():5} | Max: {df['seq_length'].max():5} | "
          f"Mean: {df['seq_length'].mean():7.1f} | Median: {df['seq_length'].median():6.0f}")

# %% [markdown]
# ### 3.3 Composition en nucléotides

# %%
def analyze_nucleotide_composition(sequences, name):
    """Analyse la composition en nucléotides d'un ensemble de séquences."""
    all_nucleotides = ''.join(sequences['sequence'].values)
    counts = Counter(all_nucleotides)
    total = sum(counts.values())

    composition = {nt: (count / total) * 100 for nt, count in counts.items()}
    return composition

# Analyser chaque dataset
compositions = {}
for name, df in [('Train', train_sequences), ('Validation', validation_sequences), ('Test', test_sequences)]:
    compositions[name] = analyze_nucleotide_composition(df, name)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, comp) in enumerate(compositions.items()):
    nucleotides = ['A', 'U', 'G', 'C']
    values = [comp.get(nt, 0) for nt in nucleotides]
    colors = [NUCLEOTIDE_COLORS[nt] for nt in nucleotides]

    bars = axes[idx].bar(nucleotides, values, color=colors, edgecolor='black', linewidth=1.5)
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Pourcentage (%)')
    axes[idx].set_ylim(0, 35)

    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, values):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('🧬 Composition en Nucléotides (A, U, G, C)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Autres nucléotides (modifications)
print("\n🔬 Nucléotides non-canoniques détectés:")
for name, comp in compositions.items():
    non_canonical = {k: v for k, v in comp.items() if k not in ['A', 'U', 'G', 'C']}
    if non_canonical:
        print(f"  {name}: {non_canonical}")
    else:
        print(f"  {name}: Aucun")

# %% [markdown]
# ### 3.4 Distribution temporelle (temporal_cutoff)

# %%
# Convertir les dates
train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
validation_sequences['temporal_cutoff'] = pd.to_datetime(validation_sequences['temporal_cutoff'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
train_sequences.groupby(train_sequences['temporal_cutoff'].dt.to_period('M')).size().plot(
    kind='bar', ax=axes[0], color='steelblue', alpha=0.7
)
axes[0].set_title('Train - Distribution temporelle', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date de publication')
axes[0].set_ylabel('Nombre de structures')
axes[0].tick_params(axis='x', rotation=45)

# Validation
validation_sequences.groupby(validation_sequences['temporal_cutoff'].dt.to_period('M')).size().plot(
    kind='bar', ax=axes[1], color='seagreen', alpha=0.7
)
axes[1].set_title('Validation - Distribution temporelle', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date de publication')
axes[1].set_ylabel('Nombre de structures')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('📅 Distribution Temporelle des Structures', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"\n📅 Plages temporelles:")
print(f"  Train:      {train_sequences['temporal_cutoff'].min().date()} → {train_sequences['temporal_cutoff'].max().date()}")
print(f"  Validation: {validation_sequences['temporal_cutoff'].min().date()} → {validation_sequences['temporal_cutoff'].max().date()}")

# %% [markdown]
# ### 3.5 Analyse de la stoichiométrie

# %%
# Analyser les patterns de stoichiométrie
def parse_stoichiometry(stoich_str):
    """Parse stoichiometry string to extract chain counts."""
    if pd.isna(stoich_str):
        return {}
    chains = {}
    for part in stoich_str.split(';'):
        if ':' in part:
            chain, count = part.split(':')
            chains[chain.strip()] = int(count)
    return chains

train_sequences['n_chains'] = train_sequences['stoichiometry'].apply(
    lambda x: sum(parse_stoichiometry(x).values()) if pd.notna(x) else 0
)
validation_sequences['n_chains'] = validation_sequences['stoichiometry'].apply(
    lambda x: sum(parse_stoichiometry(x).values()) if pd.notna(x) else 0
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribution du nombre de chaînes
for idx, (name, df, color) in enumerate([('Train', train_sequences, 'steelblue'),
                                          ('Validation', validation_sequences, 'seagreen')]):
    chain_counts = df['n_chains'].value_counts().sort_index()
    axes[idx].bar(chain_counts.index, chain_counts.values, color=color, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{name} - Nombre de chaînes', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Nombre de chaînes')
    axes[idx].set_ylabel('Fréquence')

plt.suptitle('🔗 Distribution du Nombre de Chaînes par Structure', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n📊 Statistiques sur les chaînes:")
print(f"  Train - Moyenne: {train_sequences['n_chains'].mean():.2f}, Max: {train_sequences['n_chains'].max()}")
print(f"  Validation - Moyenne: {validation_sequences['n_chains'].mean():.2f}, Max: {validation_sequences['n_chains'].max()}")

# %% [markdown]
# ## 🎯 4. Analyse des Labels (Coordonnées 3D)

# %% [markdown]
# ### 4.1 Structure des labels

# %%
print("📋 Colonnes train_labels:")
print(train_labels.columns.tolist())
print("\n" + "=" * 60)
train_labels.head(10)

# %%
# Nombre de structures alternatives par cible
coord_cols = [c for c in train_labels.columns if c.startswith('x_')]
n_structures = len(coord_cols)
print(f"\n🔹 Nombre de structures alternatives dans train_labels: {n_structures}")
print(f"   Colonnes de coordonnées: {coord_cols}")

# Vérifier les valeurs manquantes
print(f"\n🔹 Valeurs manquantes dans les coordonnées:")
for col in ['x_1', 'y_1', 'z_1']:
    missing = train_labels[col].isna().sum()
    print(f"   {col}: {missing:,} ({missing/len(train_labels)*100:.2f}%)")

# %% [markdown]
# ### 4.2 Distribution des résidus par structure

# %%
# Compter les résidus par target_id
train_labels['target_id'] = train_labels['ID'].str.rsplit('_', n=1).str[0]
residues_per_target = train_labels.groupby('target_id').size()

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(residues_per_target, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax.set_title('Distribution du Nombre de Résidus par Structure', fontsize=12, fontweight='bold')
ax.set_xlabel('Nombre de résidus')
ax.set_ylabel('Fréquence')
ax.axvline(residues_per_target.median(), color='red', linestyle='--',
           label=f"Médiane: {residues_per_target.median():.0f}")
ax.legend()
plt.tight_layout()
plt.show()

print(f"\n📈 Statistiques des résidus par structure:")
print(f"   Min: {residues_per_target.min()}, Max: {residues_per_target.max()}")
print(f"   Mean: {residues_per_target.mean():.1f}, Median: {residues_per_target.median():.0f}")

# %% [markdown]
# ### 4.3 Distribution des types de résidus

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, df, color) in enumerate([('Train', train_labels, 'steelblue'),
                                          ('Validation', validation_labels, 'seagreen')]):
    resname_counts = df['resname'].value_counts()
    colors = [NUCLEOTIDE_COLORS.get(nt, 'gray') for nt in resname_counts.index]

    bars = axes[idx].bar(resname_counts.index, resname_counts.values, color=colors, edgecolor='black')
    axes[idx].set_title(f'{name} - Distribution des résidus', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Type de résidu')
    axes[idx].set_ylabel('Fréquence')

    # Ajouter pourcentages
    total = resname_counts.sum()
    for bar, val in zip(bars, resname_counts.values):
        pct = val / total * 100
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                      f'{pct:.1f}%', ha='center', fontsize=9)

plt.suptitle('🧬 Distribution des Types de Résidus', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Distribution spatiale des coordonnées

# %%
# Échantillonner pour la visualisation
sample_labels = train_labels.dropna(subset=['x_1', 'y_1', 'z_1']).sample(min(50000, len(train_labels)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (coord, color) in enumerate([('x_1', 'red'), ('y_1', 'green'), ('z_1', 'blue')]):
    axes[idx].hist(sample_labels[coord], bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'Distribution {coord.upper()}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(f'{coord} (Ångströms)')
    axes[idx].set_ylabel('Fréquence')

    mean_val = sample_labels[coord].mean()
    axes[idx].axvline(mean_val, color='black', linestyle='--', label=f'Mean: {mean_val:.1f}Å')
    axes[idx].legend()

plt.suptitle('📐 Distribution Spatiale des Coordonnées C1\'', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🌐 5. Visualisation 3D des Structures RNA

# %% [markdown]
# ### 5.1 Fonction de visualisation 3D

# %%
def visualize_rna_3d(labels_df, target_id, structure_idx=1, title=None):
    """
    Visualise une structure RNA en 3D avec Plotly.

    Args:
        labels_df: DataFrame des labels
        target_id: ID de la cible à visualiser
        structure_idx: Index de la structure (1, 2, etc.)
        title: Titre personnalisé
    """
    # Filtrer pour cette cible
    target_data = labels_df[labels_df['ID'].str.startswith(target_id + '_')].copy()
    target_data = target_data.sort_values('resid')

    # Colonnes de coordonnées
    x_col, y_col, z_col = f'x_{structure_idx}', f'y_{structure_idx}', f'z_{structure_idx}'

    # Vérifier que les colonnes existent
    if x_col not in target_data.columns:
        print(f"Structure {structure_idx} non disponible pour {target_id}")
        return None

    # Retirer les valeurs manquantes
    target_data = target_data.dropna(subset=[x_col, y_col, z_col])

    if len(target_data) == 0:
        print(f"Pas de données pour {target_id}")
        return None

    # Couleurs par nucléotide
    colors = [NUCLEOTIDE_COLORS.get(res, 'gray') for res in target_data['resname']]

    # Créer la figure
    fig = go.Figure()

    # Ajouter la trace du backbone (ligne)
    fig.add_trace(go.Scatter3d(
        x=target_data[x_col],
        y=target_data[y_col],
        z=target_data[z_col],
        mode='lines',
        line=dict(color='lightgray', width=3),
        name='Backbone',
        hoverinfo='skip'
    ))

    # Ajouter les points (atomes C1')
    fig.add_trace(go.Scatter3d(
        x=target_data[x_col],
        y=target_data[y_col],
        z=target_data[z_col],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            opacity=0.9,
            line=dict(color='black', width=0.5)
        ),
        text=[f"Res {row['resid']}: {row['resname']}<br>({row[x_col]:.2f}, {row[y_col]:.2f}, {row[z_col]:.2f})"
              for _, row in target_data.iterrows()],
        hoverinfo='text',
        name="Résidus C1'"
    ))

    # Mise en page
    title_text = title or f"Structure 3D: {target_id} (n={len(target_data)} résidus)"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data'
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    # Ajouter légende des couleurs
    for nt, color in NUCLEOTIDE_COLORS.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=f'{nt}',
            showlegend=True
        ))

    return fig

# %% [markdown]
# ### 5.2 Visualisation de structures exemple

# %%
# Sélectionner quelques structures intéressantes
target_ids = train_labels['ID'].str.rsplit('_', n=1).str[0].unique()

# Trouver des structures de différentes tailles
sizes = train_labels.groupby(train_labels['ID'].str.rsplit('_', n=1).str[0]).size()
small_target = sizes[sizes < 50].index[0] if len(sizes[sizes < 50]) > 0 else sizes.index[0]
medium_target = sizes[(sizes >= 50) & (sizes < 150)].index[0] if len(sizes[(sizes >= 50) & (sizes < 150)]) > 0 else sizes.index[1]
large_target = sizes[sizes >= 150].index[0] if len(sizes[sizes >= 150]) > 0 else sizes.index[2]

print(f"🔍 Structures sélectionnées pour visualisation:")
print(f"   Petite:  {small_target} ({sizes[small_target]} résidus)")
print(f"   Moyenne: {medium_target} ({sizes[medium_target]} résidus)")
print(f"   Grande:  {large_target} ({sizes[large_target]} résidus)")

# %%
# Visualiser une petite structure
fig = visualize_rna_3d(train_labels, small_target)
if fig:
    fig.show()

# %%
# Visualiser une structure moyenne
fig = visualize_rna_3d(train_labels, medium_target)
if fig:
    fig.show()

# %%
# Visualiser une grande structure
fig = visualize_rna_3d(train_labels, large_target)
if fig:
    fig.show()

# %% [markdown]
# ### 5.3 Comparaison de structures alternatives (si disponibles)

# %%
def compare_structures(labels_df, target_id, n_structures=2):
    """Compare plusieurs structures alternatives pour une même cible."""
    target_data = labels_df[labels_df['ID'].str.startswith(target_id + '_')].copy()
    target_data = target_data.sort_values('resid')

    # Vérifier combien de structures sont disponibles
    x_cols = [c for c in target_data.columns if c.startswith('x_')]
    n_available = len(x_cols)

    if n_available < 2:
        print(f"Seulement {n_available} structure(s) disponible(s) pour {target_id}")
        return None

    n_to_show = min(n_structures, n_available)

    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i in range(1, n_to_show + 1):
        x_col, y_col, z_col = f'x_{i}', f'y_{i}', f'z_{i}'
        data = target_data.dropna(subset=[x_col, y_col, z_col])

        if len(data) > 0:
            fig.add_trace(go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col],
                mode='lines+markers',
                marker=dict(size=4, opacity=0.7),
                line=dict(width=2),
                name=f'Structure {i}',
                marker_color=colors[i-1]
            ))

    fig.update_layout(
        title=f"Comparaison des structures: {target_id}",
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data'
        ),
        width=800,
        height=600
    )

    return fig

# Trouver une cible avec plusieurs structures
multi_struct_cols = [c for c in train_labels.columns if c.startswith('x_')]
if len(multi_struct_cols) > 1:
    # Vérifier quelles cibles ont plusieurs conformations
    sample_target = train_labels['ID'].str.rsplit('_', n=1).str[0].iloc[0]
    fig = compare_structures(train_labels, sample_target)
    if fig:
        fig.show()

# %% [markdown]
# ### 5.4 Analyse des distances inter-résidus

# %%
def analyze_distances(labels_df, target_id, structure_idx=1):
    """Analyse les distances entre résidus consécutifs."""
    target_data = labels_df[labels_df['ID'].str.startswith(target_id + '_')].copy()
    target_data = target_data.sort_values('resid')

    x_col, y_col, z_col = f'x_{structure_idx}', f'y_{structure_idx}', f'z_{structure_idx}'
    target_data = target_data.dropna(subset=[x_col, y_col, z_col])

    if len(target_data) < 2:
        return None

    # Calculer les distances consécutives
    coords = target_data[[x_col, y_col, z_col]].values
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))

    return distances

# Calculer pour plusieurs structures
all_distances = []
sample_targets = train_labels['ID'].str.rsplit('_', n=1).str[0].unique()[:100]

for target in sample_targets:
    distances = analyze_distances(train_labels, target)
    if distances is not None:
        all_distances.extend(distances)

all_distances = np.array(all_distances)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
axes[0].hist(all_distances, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0].set_title('Distribution des Distances C1\'-C1\' Consécutives', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Distance (Å)')
axes[0].set_ylabel('Fréquence')
axes[0].axvline(np.median(all_distances), color='red', linestyle='--',
                label=f'Médiane: {np.median(all_distances):.2f}Å')
axes[0].legend()

# Box plot
axes[1].boxplot(all_distances, vert=True)
axes[1].set_title('Box Plot des Distances', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Distance (Å)')

plt.suptitle('📏 Analyse des Distances Inter-Résidus', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"\n📊 Statistiques des distances C1'-C1' consécutives:")
print(f"   Mean: {np.mean(all_distances):.2f}Å")
print(f"   Median: {np.median(all_distances):.2f}Å")
print(f"   Std: {np.std(all_distances):.2f}Å")
print(f"   Min: {np.min(all_distances):.2f}Å, Max: {np.max(all_distances):.2f}Å")

# %% [markdown]
# ## 📂 6. Analyse des MSA (Multiple Sequence Alignments)

# %%
def analyze_msa(msa_path, n_samples=5):
    """Analyse un fichier MSA."""
    try:
        sequences = parse_fasta(msa_path)

        info = {
            'n_sequences': len(sequences),
            'target_length': len(sequences[0][1]) if sequences else 0,
            'headers': [seq[0] for seq in sequences[:n_samples]]
        }

        return info
    except Exception as e:
        return {'error': str(e)}

# Analyser quelques MSA
msa_dir = DATA_PATH / 'MSA'
msa_files = list(msa_dir.glob('*.fasta'))[:10]

print(f"📁 Analyse de {len(msa_files)} fichiers MSA (sur {len(list(msa_dir.glob('*.fasta')))} total):\n")

msa_stats = []
for msa_file in msa_files:
    info = analyze_msa(msa_file)
    msa_stats.append({
        'file': msa_file.name,
        'n_sequences': info.get('n_sequences', 0),
        'target_length': info.get('target_length', 0)
    })
    print(f"  {msa_file.name}: {info.get('n_sequences', 'N/A')} séquences, longueur {info.get('target_length', 'N/A')}")

# %%
# Distribution de la profondeur des MSA
all_msa_files = list(msa_dir.glob('*.fasta'))
msa_depths = []

print(f"\n⏳ Analyse de la profondeur des MSA ({len(all_msa_files)} fichiers)...")

for i, msa_file in enumerate(all_msa_files[:500]):  # Limiter pour la vitesse
    try:
        sequences = parse_fasta(msa_file)
        msa_depths.append(len(sequences))
    except:
        pass

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(msa_depths, bins=50, color='teal', alpha=0.7, edgecolor='black')
ax.set_title('Distribution de la Profondeur des MSA', fontsize=12, fontweight='bold')
ax.set_xlabel('Nombre de séquences dans le MSA')
ax.set_ylabel('Fréquence')
ax.axvline(np.median(msa_depths), color='red', linestyle='--', label=f'Médiane: {np.median(msa_depths):.0f}')
ax.legend()
plt.tight_layout()
plt.show()

print(f"\n📊 Statistiques de profondeur MSA:")
print(f"   Min: {np.min(msa_depths)}, Max: {np.max(msa_depths)}")
print(f"   Mean: {np.mean(msa_depths):.1f}, Median: {np.median(msa_depths):.0f}")

# %% [markdown]
# ## 📋 7. Analyse des Métadonnées

# %%
print("📋 Colonnes rna_metadata:")
print(rna_metadata.columns.tolist())
print(f"\nShape: {rna_metadata.shape}")
rna_metadata.head()

# %%
# Analyse des colonnes clés
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Resolution
if 'resolution' in rna_metadata.columns:
    rna_metadata['resolution'].dropna().hist(bins=50, ax=axes[0, 0], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Distribution de la Résolution', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Résolution (Å)')

# Method
if 'method' in rna_metadata.columns:
    method_counts = rna_metadata['method'].value_counts().head(10)
    method_counts.plot(kind='barh', ax=axes[0, 1], color='seagreen', alpha=0.7)
    axes[0, 1].set_title('Méthodes Expérimentales', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Nombre de structures')

# RNA composition
if 'rna_composition' in rna_metadata.columns:
    rna_metadata['rna_composition'].dropna().hist(bins=50, ax=axes[1, 0], color='coral', alpha=0.7)
    axes[1, 0].set_title('Composition RNA (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('% RNA')

# Structuredness
if 'structuredness' in rna_metadata.columns:
    rna_metadata['structuredness'].dropna().hist(bins=50, ax=axes[1, 1], color='purple', alpha=0.7)
    axes[1, 1].set_title('Structuredness', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Structuredness score')

plt.suptitle('📊 Analyse des Métadonnées RNA', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 📝 8. Analyse du Format de Soumission

# %%
print("📋 Format de soumission:")
print(sample_submission.columns.tolist())
print(f"\nShape: {sample_submission.shape}")
sample_submission.head(10)

# %%
# Vérifier le nombre de prédictions requises
coord_cols = [c for c in sample_submission.columns if c.startswith('x_')]
print(f"\n🎯 Nombre de structures à prédire: {len(coord_cols)} (x_1 à x_{len(coord_cols)})")

# Nombre de cibles uniques dans le test
test_target_ids = sample_submission['ID'].str.rsplit('_', n=1).str[0].unique()
print(f"📊 Nombre de cibles test: {len(test_target_ids)}")
print(f"📊 Nombre total de résidus à prédire: {len(sample_submission)}")

# %% [markdown]
# ## 🎯 9. Résumé et Insights Clés

# %%
print("=" * 70)
print("                    📊 RÉSUMÉ DE L'ANALYSE EDA")
print("=" * 70)

print("""
🔹 DONNÉES:
   • Train: {train_seq} séquences, {train_res:,} résidus
   • Validation: {val_seq} séquences, {val_res:,} résidus
   • Test: {test_seq} séquences à prédire
   • MSA disponibles: {msa_count:,} fichiers
   • Structures PDB: {pdb_count:,} fichiers

🔹 SÉQUENCES:
   • Longueur médiane: Train={train_med:.0f}, Val={val_med:.0f}, Test={test_med:.0f}
   • Composition: ~25% chaque nucléotide (A, U, G, C équilibrés)
   • Structures multi-chaînes présentes

🔹 STRUCTURES 3D:
   • Coordonnées: Position de l'atome C1' de chaque résidu
   • Distance C1'-C1' consécutive: ~{dist_mean:.1f}Å (médiane)
   • Format soumission: 5 structures par cible

🔹 INSIGHTS POUR LA MODÉLISATION:
   • Les MSA profonds peuvent aider (info évolutive)
   • Templates PDB disponibles pour recherche d'homologues
   • Validation temporelle: structures après mai 2025
   • Métrique: TM-score (best of 5 predictions)
""".format(
    train_seq=len(train_sequences),
    train_res=len(train_labels),
    val_seq=len(validation_sequences),
    val_res=len(validation_labels),
    test_seq=len(test_sequences),
    msa_count=len(list((DATA_PATH / 'MSA').glob('*.fasta'))),
    pdb_count=len(list((DATA_PATH / 'PDB_RNA').glob('*.cif'))),
    train_med=train_sequences['seq_length'].median(),
    val_med=validation_sequences['seq_length'].median(),
    test_med=test_sequences['seq_length'].median(),
    dist_mean=np.median(all_distances) if len(all_distances) > 0 else 5.9
))

print("=" * 70)
print("                    🚀 PRÊT POUR LA MODÉLISATION!")
print("=" * 70)
