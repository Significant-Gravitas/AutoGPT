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

# %% [markdown]
# ---
# # 🧬 PARTIE 2 : STRATÉGIES DE MODÉLISATION
# ---
#
# Basé sur les solutions gagnantes de la Part 1 :
# - **1ère place (john)** : Template-Based Modeling (TBM) pur
# - **2ème place (odat)** : TBM avec alignement optimisé
# - **3ème place (Eigen)** : Hybride TBM + Deep Learning
#
# ## Approche retenue : Template-Based Modeling
# 95% des cibles ont des templates potentiels dans le PDB !

# %% [markdown]
# ## 🔍 10. Pipeline Template-Based Modeling (TBM)

# %%
# ============================================
# STRATÉGIE 1 : TEMPLATE-BASED MODELING
# ============================================
# Pipeline :
# 1. Parser les séquences du PDB_RNA
# 2. Aligner la séquence cible avec les séquences PDB
# 3. Trouver les meilleurs templates
# 4. Copier/adapter les coordonnées 3D
# 5. Générer 5 prédictions diversifiées

print("🔧 Configuration du pipeline TBM...")

# %% [markdown]
# ### 10.1 Extraction des séquences du PDB

# %%
from gemmi import cif  # Parser CIF rapide (disponible sur Kaggle)

def extract_rna_sequences_from_cif(cif_path):
    """
    Extrait les séquences RNA et coordonnées C1' d'un fichier CIF.
    Retourne un dict avec les infos de la structure.
    """
    try:
        doc = cif.read(str(cif_path))
        block = doc.sole_block()

        # Extraire les coordonnées des atomes
        atom_site = block.find('_atom_site.',
            ['label_atom_id', 'label_comp_id', 'label_asym_id',
             'label_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
             'type_symbol', 'group_PDB'])

        if not atom_site:
            return None

        residues = {}
        coords = {}

        for row in atom_site:
            atom_name = row[0]
            res_name = row[1]
            chain_id = row[2]
            seq_id = row[3]

            # Filtrer uniquement les nucléotides RNA (A, U, G, C)
            if res_name in ['A', 'U', 'G', 'C', 'ADE', 'URA', 'GUA', 'CYT']:
                # Mapper les noms longs vers courts
                res_short = {'ADE': 'A', 'URA': 'U', 'GUA': 'G', 'CYT': 'C'}.get(res_name, res_name)

                key = (chain_id, seq_id)
                if key not in residues:
                    residues[key] = res_short

                # Stocker les coordonnées C1'
                if atom_name == "C1'":
                    try:
                        x, y, z = float(row[4]), float(row[5]), float(row[6])
                        coords[key] = (x, y, z)
                    except:
                        pass

        # Construire la séquence par chaîne
        chains = {}
        for (chain_id, seq_id), res in sorted(residues.items()):
            if chain_id not in chains:
                chains[chain_id] = {'sequence': '', 'coords': []}
            chains[chain_id]['sequence'] += res
            if (chain_id, seq_id) in coords:
                chains[chain_id]['coords'].append(coords[(chain_id, seq_id)])
            else:
                chains[chain_id]['coords'].append(None)

        return {
            'pdb_id': cif_path.stem.upper(),
            'chains': chains
        }

    except Exception as e:
        return None

# Test sur quelques fichiers
print("📂 Test d'extraction sur quelques fichiers CIF...")
pdb_dir = DATA_PATH / 'PDB_RNA'
sample_cifs = list(pdb_dir.glob('*.cif'))[:5]

for cif_file in sample_cifs:
    result = extract_rna_sequences_from_cif(cif_file)
    if result:
        for chain_id, chain_data in result['chains'].items():
            seq = chain_data['sequence']
            n_coords = sum(1 for c in chain_data['coords'] if c is not None)
            print(f"  {result['pdb_id']}_{chain_id}: {len(seq)} nt, {n_coords} coords C1'")
            if len(seq) < 50:
                print(f"    Séquence: {seq}")

# %% [markdown]
# ### 10.2 Construction de la base de templates

# %%
def build_template_database(pdb_dir, max_files=None, verbose=True):
    """
    Construit une base de données de templates à partir des fichiers CIF.
    """
    templates = []
    cif_files = list(pdb_dir.glob('*.cif'))

    if max_files:
        cif_files = cif_files[:max_files]

    if verbose:
        print(f"📦 Construction de la base de templates ({len(cif_files)} fichiers)...")

    for i, cif_file in enumerate(cif_files):
        if verbose and i % 500 == 0:
            print(f"   Progression: {i}/{len(cif_files)}")

        result = extract_rna_sequences_from_cif(cif_file)
        if result:
            for chain_id, chain_data in result['chains'].items():
                seq = chain_data['sequence']
                coords = chain_data['coords']

                # Filtrer les chaînes trop courtes ou sans coordonnées
                n_valid_coords = sum(1 for c in coords if c is not None)
                if len(seq) >= 10 and n_valid_coords >= len(seq) * 0.5:
                    templates.append({
                        'pdb_id': result['pdb_id'],
                        'chain_id': chain_id,
                        'sequence': seq,
                        'coords': coords,
                        'length': len(seq)
                    })

    if verbose:
        print(f"✅ {len(templates)} templates extraits")

    return templates

# Construire une base de templates (limiter pour la démo)
# En production, utiliser tous les fichiers (max_files=None)
templates_db = build_template_database(pdb_dir, max_files=500, verbose=True)

# Statistiques
template_lengths = [t['length'] for t in templates_db]
print(f"\n📊 Statistiques des templates:")
print(f"   Nombre: {len(templates_db)}")
print(f"   Longueur min: {min(template_lengths)}, max: {max(template_lengths)}")
print(f"   Longueur moyenne: {np.mean(template_lengths):.1f}")

# %% [markdown]
# ### 10.3 Alignement de séquences (Simple)

# %%
def simple_sequence_alignment(seq1, seq2):
    """
    Alignement simple basé sur la similarité de séquence.
    Retourne le score de similarité (0-1) et les positions alignées.
    """
    # Méthode simple : recherche de sous-chaînes communes
    len1, len2 = len(seq1), len(seq2)

    if len1 == 0 or len2 == 0:
        return 0.0, []

    # Construire une matrice de correspondance
    matches = 0
    aligned_positions = []

    # Alignement global simple (sans gaps pour commencer)
    min_len = min(len1, len2)

    # Essayer différents décalages
    best_score = 0
    best_offset = 0

    for offset in range(-len2 + 1, len1):
        score = 0
        for i in range(min_len):
            pos1 = i
            pos2 = i - offset
            if 0 <= pos1 < len1 and 0 <= pos2 < len2:
                if seq1[pos1] == seq2[pos2]:
                    score += 1

        if score > best_score:
            best_score = score
            best_offset = offset

    # Reconstruire l'alignement avec le meilleur offset
    for i in range(max(len1, len2)):
        pos1 = i
        pos2 = i - best_offset
        if 0 <= pos1 < len1 and 0 <= pos2 < len2:
            if seq1[pos1] == seq2[pos2]:
                aligned_positions.append((pos1, pos2))

    # Score normalisé
    similarity = best_score / max(len1, len2)

    return similarity, aligned_positions

def find_best_templates(query_seq, templates_db, top_k=5):
    """
    Trouve les meilleurs templates pour une séquence query.
    """
    scores = []

    for template in templates_db:
        similarity, aligned_pos = simple_sequence_alignment(query_seq, template['sequence'])

        # Bonus pour longueur similaire
        len_ratio = min(len(query_seq), template['length']) / max(len(query_seq), template['length'])

        combined_score = similarity * 0.7 + len_ratio * 0.3

        scores.append({
            'template': template,
            'similarity': similarity,
            'len_ratio': len_ratio,
            'combined_score': combined_score,
            'aligned_positions': aligned_pos
        })

    # Trier par score combiné
    scores.sort(key=lambda x: x['combined_score'], reverse=True)

    return scores[:top_k]

# Test sur une séquence de validation
print("🔍 Test de recherche de templates...")
test_seq = validation_sequences.iloc[0]['sequence']
print(f"   Séquence test: {test_seq[:50]}... (longueur: {len(test_seq)})")

best_templates = find_best_templates(test_seq, templates_db, top_k=5)

print(f"\n📋 Top 5 templates:")
for i, match in enumerate(best_templates):
    t = match['template']
    print(f"   {i+1}. {t['pdb_id']}_{t['chain_id']}: "
          f"sim={match['similarity']:.3f}, len={t['length']}, "
          f"score={match['combined_score']:.3f}")

# %% [markdown]
# ### 10.4 Prédiction de structure par template

# %%
def predict_structure_from_template(query_seq, template_match, noise_std=0.0):
    """
    Prédit les coordonnées 3D en utilisant un template.

    Args:
        query_seq: Séquence cible
        template_match: Résultat de find_best_templates
        noise_std: Écart-type du bruit à ajouter (pour diversification)

    Returns:
        Liste de coordonnées (x, y, z) pour chaque résidu
    """
    template = template_match['template']
    aligned_positions = template_match['aligned_positions']
    template_coords = template['coords']

    # Initialiser les coordonnées prédites
    predicted_coords = [(0.0, 0.0, 0.0)] * len(query_seq)

    # Copier les coordonnées des positions alignées
    for query_pos, template_pos in aligned_positions:
        if template_pos < len(template_coords) and template_coords[template_pos] is not None:
            x, y, z = template_coords[template_pos]

            # Ajouter du bruit pour diversification
            if noise_std > 0:
                x += np.random.normal(0, noise_std)
                y += np.random.normal(0, noise_std)
                z += np.random.normal(0, noise_std)

            predicted_coords[query_pos] = (x, y, z)

    # Interpoler les positions manquantes
    predicted_coords = interpolate_missing_coords(predicted_coords)

    return predicted_coords

def interpolate_missing_coords(coords):
    """
    Interpole les coordonnées manquantes (0, 0, 0) par interpolation linéaire.
    """
    n = len(coords)
    result = list(coords)

    # Trouver les positions avec des coordonnées valides
    valid_indices = [i for i, c in enumerate(coords) if c != (0.0, 0.0, 0.0)]

    if len(valid_indices) < 2:
        # Pas assez de points pour interpoler, utiliser des coordonnées par défaut
        for i in range(n):
            if result[i] == (0.0, 0.0, 0.0):
                # Position sur une hélice simple
                result[i] = (i * 5.9, 0.0, 0.0)
        return result

    # Interpoler entre les points valides
    for i in range(n):
        if result[i] == (0.0, 0.0, 0.0):
            # Trouver les voisins valides les plus proches
            prev_valid = None
            next_valid = None

            for vi in valid_indices:
                if vi < i:
                    prev_valid = vi
                elif vi > i and next_valid is None:
                    next_valid = vi
                    break

            if prev_valid is not None and next_valid is not None:
                # Interpolation linéaire
                t = (i - prev_valid) / (next_valid - prev_valid)
                x = coords[prev_valid][0] + t * (coords[next_valid][0] - coords[prev_valid][0])
                y = coords[prev_valid][1] + t * (coords[next_valid][1] - coords[prev_valid][1])
                z = coords[prev_valid][2] + t * (coords[next_valid][2] - coords[prev_valid][2])
                result[i] = (x, y, z)
            elif prev_valid is not None:
                # Extrapolation depuis le dernier point valide
                dx = 5.9  # Distance moyenne C1'-C1'
                result[i] = (coords[prev_valid][0] + dx * (i - prev_valid),
                           coords[prev_valid][1],
                           coords[prev_valid][2])
            elif next_valid is not None:
                # Extrapolation avant le premier point valide
                dx = 5.9
                result[i] = (coords[next_valid][0] - dx * (next_valid - i),
                           coords[next_valid][1],
                           coords[next_valid][2])

    return result

# Test de prédiction
print("🧬 Test de prédiction de structure...")
if best_templates:
    pred_coords = predict_structure_from_template(test_seq, best_templates[0])
    valid_coords = sum(1 for c in pred_coords if c != (0.0, 0.0, 0.0))
    print(f"   Coordonnées prédites: {len(pred_coords)} résidus, {valid_coords} valides")
    print(f"   Premiers résidus: {pred_coords[:3]}")

# %% [markdown]
# ### 10.5 Génération de 5 prédictions diversifiées

# %%
def generate_diverse_predictions(query_seq, templates_db, n_predictions=5):
    """
    Génère 5 prédictions diversifiées pour une séquence.

    Stratégies de diversification :
    1. Utiliser les top-5 templates différents
    2. Ajouter du bruit aux coordonnées
    3. Combiner plusieurs templates
    """
    predictions = []

    # Trouver les meilleurs templates
    best_templates = find_best_templates(query_seq, templates_db, top_k=10)

    if not best_templates:
        # Fallback : prédiction linéaire
        for i in range(n_predictions):
            coords = [(j * 5.9 + np.random.normal(0, 0.5),
                      np.random.normal(0, 1),
                      np.random.normal(0, 1)) for j in range(len(query_seq))]
            predictions.append(coords)
        return predictions

    # Stratégie 1-3 : Utiliser les 3 meilleurs templates
    for i in range(min(3, len(best_templates))):
        noise = i * 0.2  # Augmenter le bruit progressivement
        coords = predict_structure_from_template(query_seq, best_templates[i], noise_std=noise)
        predictions.append(coords)

    # Stratégie 4 : Moyenne des 2 meilleurs templates + bruit
    if len(best_templates) >= 2:
        coords1 = predict_structure_from_template(query_seq, best_templates[0])
        coords2 = predict_structure_from_template(query_seq, best_templates[1])

        avg_coords = []
        for c1, c2 in zip(coords1, coords2):
            avg = ((c1[0] + c2[0]) / 2 + np.random.normal(0, 0.3),
                   (c1[1] + c2[1]) / 2 + np.random.normal(0, 0.3),
                   (c1[2] + c2[2]) / 2 + np.random.normal(0, 0.3))
            avg_coords.append(avg)
        predictions.append(avg_coords)
    else:
        predictions.append(predict_structure_from_template(query_seq, best_templates[0], noise_std=0.5))

    # Stratégie 5 : Meilleur template avec perturbation aléatoire
    coords = predict_structure_from_template(query_seq, best_templates[0], noise_std=0.8)
    predictions.append(coords)

    # S'assurer qu'on a exactement 5 prédictions
    while len(predictions) < n_predictions:
        predictions.append(predict_structure_from_template(query_seq, best_templates[0],
                                                          noise_std=np.random.uniform(0.3, 1.0)))

    return predictions[:n_predictions]

# Test
print("🎯 Génération de 5 prédictions diversifiées...")
test_predictions = generate_diverse_predictions(test_seq, templates_db, n_predictions=5)
print(f"   Nombre de prédictions: {len(test_predictions)}")
for i, pred in enumerate(test_predictions):
    print(f"   Prédiction {i+1}: {len(pred)} résidus")

# %% [markdown]
# ## 📊 11. Évaluation locale (TM-score simplifié)

# %%
def calculate_tm_score_simple(pred_coords, ref_coords):
    """
    Calcule un TM-score simplifié entre les coordonnées prédites et de référence.
    Note: Version simplifiée sans l'optimisation de rotation/translation.
    """
    n = len(ref_coords)
    if n == 0:
        return 0.0

    # Calculer d0
    if n >= 30:
        d0 = 0.6 * (n - 0.5) ** (1/3) - 2.5
    else:
        d0_values = {12: 0.3, 15: 0.4, 19: 0.5, 23: 0.6, 29: 0.7}
        d0 = 0.3
        for threshold, value in d0_values.items():
            if n >= threshold:
                d0 = value

    d0 = max(d0, 0.5)  # Minimum d0

    # Centrer les coordonnées
    pred_arr = np.array(pred_coords)
    ref_arr = np.array(ref_coords)

    pred_centered = pred_arr - np.mean(pred_arr, axis=0)
    ref_centered = ref_arr - np.mean(ref_arr, axis=0)

    # Calculer les distances
    distances = np.sqrt(np.sum((pred_centered - ref_centered) ** 2, axis=1))

    # Calculer le TM-score
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / n

    return tm_score

# Évaluation sur le jeu de validation
print("📊 Évaluation sur le jeu de validation...")

# Prendre quelques exemples de validation
n_eval = min(10, len(validation_sequences))
eval_scores = []

for idx in range(n_eval):
    row = validation_sequences.iloc[idx]
    target_id = row['target_id']
    query_seq = row['sequence']

    # Obtenir les coordonnées de référence
    ref_data = validation_labels[validation_labels['ID'].str.startswith(target_id + '_')]
    if len(ref_data) == 0:
        continue

    ref_data = ref_data.sort_values('resid')
    ref_coords = list(zip(ref_data['x_1'].fillna(0),
                          ref_data['y_1'].fillna(0),
                          ref_data['z_1'].fillna(0)))

    # Générer les prédictions
    predictions = generate_diverse_predictions(query_seq, templates_db)

    # Calculer le meilleur TM-score (best of 5)
    best_score = 0
    for pred in predictions:
        # Ajuster la longueur si nécessaire
        if len(pred) != len(ref_coords):
            min_len = min(len(pred), len(ref_coords))
            pred = pred[:min_len]
            ref = ref_coords[:min_len]
        else:
            ref = ref_coords

        score = calculate_tm_score_simple(pred, ref)
        best_score = max(best_score, score)

    eval_scores.append({'target_id': target_id, 'tm_score': best_score, 'length': len(query_seq)})
    print(f"   {target_id}: TM-score = {best_score:.4f} (len={len(query_seq)})")

if eval_scores:
    mean_score = np.mean([s['tm_score'] for s in eval_scores])
    print(f"\n📈 TM-score moyen (validation): {mean_score:.4f}")

# %% [markdown]
# ## 📝 12. Génération du fichier de soumission

# %%
def generate_submission(test_sequences_df, templates_db, output_path='submission.csv'):
    """
    Génère le fichier de soumission au format Kaggle.
    """
    print("📝 Génération du fichier de soumission...")

    rows = []

    for idx, row in test_sequences_df.iterrows():
        target_id = row['target_id']
        query_seq = row['sequence']

        # Générer 5 prédictions
        predictions = generate_diverse_predictions(query_seq, templates_db)

        # Créer les lignes pour chaque résidu
        for resid, nucleotide in enumerate(query_seq, start=1):
            row_data = {
                'ID': f"{target_id}_{resid}",
                'resname': nucleotide,
                'resid': resid
            }

            # Ajouter les coordonnées pour chaque prédiction
            for pred_idx, pred_coords in enumerate(predictions, start=1):
                if resid - 1 < len(pred_coords):
                    x, y, z = pred_coords[resid - 1]
                else:
                    x, y, z = 0.0, 0.0, 0.0

                # Clipper les coordonnées selon les règles
                x = np.clip(x, -999.999, 9999.999)
                y = np.clip(y, -999.999, 9999.999)
                z = np.clip(z, -999.999, 9999.999)

                row_data[f'x_{pred_idx}'] = round(x, 3)
                row_data[f'y_{pred_idx}'] = round(y, 3)
                row_data[f'z_{pred_idx}'] = round(z, 3)

            rows.append(row_data)

        if idx % 10 == 0:
            print(f"   Progression: {idx + 1}/{len(test_sequences_df)}")

    # Créer le DataFrame
    submission_df = pd.DataFrame(rows)

    # Ordonner les colonnes
    coord_cols = []
    for i in range(1, 6):
        coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    column_order = ['ID', 'resname', 'resid'] + coord_cols
    submission_df = submission_df[column_order]

    # Sauvegarder
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Soumission générée: {output_path}")
    print(f"   Shape: {submission_df.shape}")

    return submission_df

# Génération de la soumission (décommentez pour exécuter)
# submission = generate_submission(test_sequences, templates_db, 'submission.csv')

# %% [markdown]
# ## 🎯 13. Résumé de la stratégie

# %%
print("=" * 70)
print("               🎯 RÉSUMÉ DE LA STRATÉGIE TBM")
print("=" * 70)
print("""
📋 PIPELINE IMPLÉMENTÉ:

1. EXTRACTION DES TEMPLATES
   • Parser les fichiers CIF du PDB_RNA
   • Extraire séquences + coordonnées C1'
   • {n_templates} templates disponibles

2. RECHERCHE DE TEMPLATES
   • Alignement de séquence simple
   • Score combiné: similarité + ratio de longueur
   • Sélection des top-K templates

3. PRÉDICTION DE STRUCTURE
   • Copie des coordonnées du template
   • Interpolation des positions manquantes
   • 5 stratégies de diversification

4. DIVERSIFICATION (5 prédictions)
   • Prédiction 1-3: Top 3 templates
   • Prédiction 4: Moyenne des 2 meilleurs
   • Prédiction 5: Perturbation aléatoire

📈 AMÉLIORATIONS POSSIBLES:
   • Alignement plus sophistiqué (Smith-Waterman)
   • Rotation/translation optimale (Kabsch algorithm)
   • Deep Learning pour les cibles sans template
   • Utilisation des MSA pour l'alignement

🚀 Pour soumettre:
   1. Décommenter la génération de soumission
   2. Utiliser tous les fichiers PDB (max_files=None)
   3. Exécuter le notebook complet
""".format(n_templates=len(templates_db)))
print("=" * 70)
