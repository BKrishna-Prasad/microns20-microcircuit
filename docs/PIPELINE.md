# Pipeline

This document describes the current workflow from MICrONS data access to the preliminary recurrent simulation.

## Overview

```text
MICrONS CAVE
    ↓
functional-coregistration and biological annotations
    ↓
candidate filtering
    ↓
CAVE skeleton retrieval and morphology checks
    ↓
recording selection
    ↓
20-neuron connectivity optimisation
    ↓
independent population checks
    ↓
simulation-ready CAVE morphologies
    ↓
synapse-to-section mapping
    ↓
structural SONATA
    ↓
structural validation
    ↓
Allen reference model
    ↓
parameter transfer to MICrONS morphologies
    ↓
20-cell independent simulation
    ↓
20-cell recurrent simulation
```

## Structural pipeline: stages 00-10

| Stage | Notebook | Main implementation | Purpose |
|---|---|---|---|
| 00 | `00_source_preflight.ipynb` | `cave.py` | Check configuration, CAVE access and required source tables |
| 01 | `01_candidate_discovery.ipynb` | `candidates.py` | Build the biological candidate pool and preserve functional mappings |
| 02 | `02_cave_morphology_eligibility.ipynb` | `morphology.py` | Retrieve CAVE skeletons and apply morphology eligibility checks |
| 03 | `03_recording_selection.ipynb` | `connectivity.py` | Compare simultaneous recordings using eligible-cell and connectivity statistics |
| 04 | `04_connectivity_selection.ipynb` | `selection.py` | Select exactly 20 cells while favouring recurrent connectivity |
| 05 | `05_final_qc_and_manifest.ipynb` | `validation.py` | Recheck identity, skeletons, mappings and connectivity, then assign model IDs |
| 06 | `06_functional_identity.ipynb` | `functional.py` | Preserve all current functional identifiers and record the deferred trace-acquisition state |
| 07 | `07_simulation_morphologies.ipynb` | `morphology.py` | Create simulation SWCs while preserving geometry and compartment labels |
| 08 | `08_synapse_mapping.ipynb` | `synapses.py` | Map observed CAVE synapses to simulation sections and positions |
| 09 | `09_structural_sonata.ipynb` | `sonata.py` | Write selected nodes, outside-selected20 nodes, edges and morphologies in SONATA |
| 10 | `10_end_to_end_validation.ipynb` | `validation.py` | Reopen and verify the structural package with multiple readers and identity checks |

### Stage 01: candidate definition

The current candidate logic starts from manual functional coregistration and then applies configured biological and proofreading rules.

The important principle is that selection rules live in `configs/project.yaml`. For example, axon proofreading strategy is configurable, so a future run can include `axon_partially_extended` in addition to the present conservative strategy without rewriting the selection code.

All functional mapping rows are retained. A neuron with more than one mapped functional unit is not silently reduced to one row.

### Stage 02: morphology eligibility

CAVE skeleton version 4 is used as the morphology source.

A morphology must contain the compartments required by the current configuration and must pass tree/topology and loader checks. Apical type-4 annotation is recorded but is not required.

The largest reduction in the candidate pool comes from the selected proofreading/axon strategy, not from the availability of cached skeletons.

### Stage 03: recording selection

Recording choice is based on the eligible neurons available within a recording and their structural network.

The current selected recording is `session 9 / scan 3`. Before the final 20-cell selection it contains 29 eligible neurons, 47 directed connected pairs and 65 individual contacts.

### Stage 04: 20-cell selection

The optimisation selects exactly 20 neurons from the selected recording.

The current 20-cell circuit contains:

```text
38 directed recurrent pairs
54 recurrent contacts
```

### Stage 05-06: identity and functional mapping

Stable nucleus, root and supervoxel identifiers are checked again after selection.

The current 20 neurons retain 22 functional mapping rows. The functional traces themselves have not yet been retrieved into this repository.

### Stage 07: simulation morphologies

The raw CAVE skeletons are never overwritten.

Four zero-radius samples were found across three selected neurons. All four were isolated terminal points. Their radii were replaced using positive topological boundary information while coordinates, topology and compartment types were left unchanged.

One selected cell (model 11, nucleus 262893) contains a proximal dendrite-to-axon type transition. Independent inspection showed one axon entry point and one connected axonal component. The source annotation is therefore preserved rather than relabelled.

### Stage 08: synapse placement

Every observed selected-to-selected synapse is retained as an individual contact.

The mapping chain links the CAVE postsynaptic site to the selected simulation morphology and then to:

```text
afferent_section_id
afferent_section_pos
```

The same approach is used for incoming contacts from presynaptic roots outside the selected 20.

Current counts:

```text
recurrent contacts:                    54
outside-selected20 incoming contacts:  44,282
outside-selected20 source roots:       29,991
```

"Outside selected 20" is deliberately not treated as synonymous with "extrinsic". Many of these sources may be local neurons that were simply not included in the 20-cell model.

### Stage 09-10: structural SONATA and checks

The structural SONATA package contains:

- selected biophysical node population;
- outside-selected20 virtual node population;
- recurrent edges;
- outside-selected20 incoming edges;
- 20 morphology files.

It does not contain fitted e-models, synaptic weights, delays or external spike trains.

The current SONATA files use explicit population information, explicit node IDs and `recenter=0` so the global MICrONS morphology coordinates are not translated again.

The current structural test suite passes 11 tests.

## Simulation sequence

The simulation work is kept separate from the structural data pipeline.

### Simulation 00: Allen reference

The original Allen morphology, fitted parameters and 16 NMODL mechanisms are run through BMTK/NEURON.

At 0.20 nA current injection the native reference model produces 16 spikes in the current development environment.

### Simulation 01: one MICrONS morphology

The Allen-derived perisomatic parameters are transferred to MICrONS model 0 while preserving the full CAVE-derived morphology.

The standard Allen perisomatic processing would replace the reconstructed axon. A project-specific cell processor therefore applies the parameter set without replacing the MICrONS axon.

Model 0 is subthreshold at 0.20 nA and produces five spikes at 0.60 nA. This difference illustrates why a parameter set fitted on one morphology should not be interpreted as a fitted model after transfer to another morphology.

### Simulation 02: all 20 cells independently

All 20 morphologies receive the same Allen-derived membrane parameters and the same 0.60 nA somatic current step.

Current result:

```text
finite simulations: 20 / 20
cells with stimulus-evoked spikes: 20 / 20
total stimulus-window spikes: 122
```

This experiment is a compatibility and morphology-response check, not cell-specific physiological validation.

### Simulation 03: recurrent circuit

The recurrent simulation reuses the validated selected-node and recurrent-edge HDF5 files.

The 54 contacts remain separate and retain their saved postsynaptic section locations.

One source cell is chosen automatically from the observed recurrent graph by:

1. number of distinct postsynaptic targets;
2. number of outgoing contacts;
3. model ID only as a deterministic tie-break.

The current source is model 13. It directly contacts models 0, 5, 6, 12 and 18 through 9 anatomical contacts.

The current simulation stimulates only model 13. Direct targets show postsynaptic depolarisations, while no unstimulated cell reaches spike threshold under the reference `Exp2Syn` parameters.

![alt text](image.png)

## BMTK 1.2 compatibility handling

Two small run-time adapters are currently needed for BMTK 1.2.0:

1. the run-local recurrent edge file exposes `node_id_to_range` in addition to the SONATA-style `node_id_to_ranges`;
2. run-local morphology filenames are decoded to normal text and placed in the node-types table so BMTK does not pass a bytes-form filename to NEURON.

These changes are applied only to simulation copies under `runs/`. The validated structural SONATA files under `data/processed/sonata/` are not modified.

These adapters should be moved to a dedicated compatibility module during the next code cleanup.
