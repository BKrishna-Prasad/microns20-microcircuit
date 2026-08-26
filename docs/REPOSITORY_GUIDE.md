# Repository guide

## Top level

```text
configs/          settings that a user is expected to change
data/             source, intermediate and processed scientific artifacts
docs/             project documentation
notebooks/        readable execution and analysis notebooks
provenance/       records showing how current pipeline outputs were produced
results/          compact tables and figures for inspection
runs/             generated simulation directories; normally not committed
scripts/          command-line helpers
src/microns20/    reusable Python code
tests/            automated checks
```

## `configs/`

### `configs/project.yaml`

Main structural-pipeline configuration.

This is where selection and data-source choices should live. Examples include:

- MICrONS datastack/materialisation;
- skeleton version;
- required areas/cell classes;
- proofreading strategy;
- number of selected neurons;
- SONATA population names.

A useful property of the current code is that selection policy can be changed without editing the notebooks. For example, a future run can broaden the allowed axon strategy from only fully extended axons to include partially extended axons, provided the downstream morphology requirements are still satisfied.

### `configs/simulations/`

One configuration per simulation experiment.

Current examples:

```text
allen_reference.yaml
microns_single_cell.yaml
microns20_independent.yaml
microns20_recurrent.yaml
```

Simulation settings should be changed here rather than directly in notebooks.

### `configs/synapses/`

Reference synaptic model parameter files.

The current `Exp2Syn` file belongs here because its kinetics are a modelling choice, not an observed MICrONS structural property.

## `data/`

### `data/raw/`

Source snapshots and CAVE skeleton cache.

Treat these files as source evidence. Pipeline code should not overwrite them in place.

This directory can become large and should normally be excluded from a lightweight public Git repository. The documentation should explain how to retrieve/rebuild it.

### `data/interim/`

Working tables passed from one stage to the next.

These are useful for debugging and complete reproduction, but they are usually unnecessary for a lightweight public code release because they can be recreated from the earlier stage.

### `data/external/`

Third-party model files that are not produced by MICrONS20.

The current Allen Cell Types model is stored locally here during development.


### `data/processed/`

The important current outputs.

Key files:

```text
final20_manifest.parquet
final20_functional_mappings.parquet
```

`final20_manifest.parquet` answers "which 20 biological neurons are in this model?"

Useful columns include:

```text
model_node_id
nucleus_id
pt_root_id
pt_supervoxel_id
session
scan_idx
microns_mtype
soma_x_um
soma_y_um
soma_z_um
```

`final20_functional_mappings.parquet` answers "which functional unit(s) correspond to each selected neuron?"

### `data/processed/morphologies/`

Contains the 20 simulation SWCs plus their manifest.

Use the manifest rather than reconstructing filenames in downstream code.

### `data/processed/connectivity/`

Important tables include:

```text
intrinsic_synapses.parquet
incoming_external_synapses.parquet
external_presynaptic_nodes.parquet
outgoing_external_synapses.parquet
unresolved_synapses.parquet
```

`intrinsic_synapses.parquet` contains the 54 selected-to-selected contacts used for the recurrent circuit.

`incoming_external_synapses.parquet` contains input from positive presynaptic roots outside the selected 20. The word "external" in historical filenames means external to the selected subset, not necessarily long-range or extrinsic to V1.

### `data/processed/sonata/`

Structural SONATA representation.

Main entry point:

```text
data/processed/sonata/circuit_config.json
```

It contains the structural network before simulation-specific physiology is added.

## `src/microns20/`

### Structural pipeline modules

| Module | Role |
|---|---|
| `config.py` | Locate the repository, load and validate configuration |
| `artifacts.py` | Table/JSON writing, hashes and schema helpers |
| `cave.py` | CAVE access and source queries |
| `candidates.py` | Candidate and functional-mapping logic |
| `connectivity.py` | Recording-level network metrics |
| `selection.py` | Exact-size connectivity optimisation |
| `morphology.py` | Skeleton checks and simulation morphology generation |
| `synapses.py` | Synapse classification and section placement |
| `sonata.py` | Structural SONATA writing |
| `functional.py` | Functional identity and acquisition status |
| `validation.py` | Population and end-to-end checks |
| `provenance.py` | Per-stage reproducibility records |
| `qc.py` | Small QC summaries |
| `orchestration.py` | Shared stage/path helpers |

### Simulation modules

Current development has introduced:

```text
simulation.py
cell_processors.py
population_simulation.py
recurrent_simulation.py
```

These work, but this part of the code can be simplified before a stable release.

Recommended cleanup:

```text
simulation/
  common.py
  cell_models.py
  independent.py
  recurrent.py
  bmtk_compat.py
```

or an equivalent compact structure.

The aim is not to create many small files for their own sake; it is to keep common run preparation, cell-model handling and BMTK-specific adapters from being duplicated.

## `notebooks/`

### `notebooks/pipeline/`

Stages 00-10.

These notebooks should remain readable records that call functions from `src/microns20/`. New scientific logic should not be duplicated inside notebooks.

### `notebooks/analysis/`

Visual inspection and presentation-oriented analysis.

The structural-circuit visualisation notebook belongs here because it is not a pipeline dependency.

Useful views include:

- one cell in 2D/3D;
- the complete 20-cell circuit in 3D;
- recurrent graph in spatial coordinates;
- graph-topology view;
- 20×20 contact matrix;
- outside-selected20 input counts;
- synapse-placement QC.

### `notebooks/simulation/`

Current sequence:

```text
00_allen_reference.ipynb
01_microns_single_cell_transfer.ipynb
02_microns20_independent_current_clamp.ipynb
03_microns20_recurrent_network.ipynb
```

The simulation notebooks should remain short once the current exploratory implementation is cleaned.

## `results/`

Use this directory for compact outputs that a person should inspect without reopening all pipeline artifacts.

Examples:

```text
results/tables/
results/figures/
```

A good public release should copy only useful summary figures/tables here rather than committing complete generated `runs/` directories.

## `runs/`

Generated BioNet simulation environments and output.

Examples:

```text
allen_reference/
microns_single_cell_model_00/
microns20_independent_060/
microns20_recurrent/
```

## `tests/`

The current structural suite has 11 passing tests.

Before calling the simulation code stable, add tests for:

- decoding morphology names for BMTK;
- recurrent edge-index compatibility;
- preservation of all 54 synapse IDs in the run copy;
- preservation of section IDs and section positions;
- correct selection of the stimulated source;
- empty-spike handling;
- voltage-report support for both `index_pointer` and `index_pointers`.

## `provenance/`

See `DATA_AND_PROVENANCE.md` for the recommended public layout.
