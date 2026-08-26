# Simulation

## Purpose

The simulation branch was added to establish that the selected MICrONS structural circuit can be instantiated in BioNet/NEURON and to identify what remains necessary before biological validation.

It is intentionally separate from the structural pipeline.

## Software environment used during development

Current recorded versions include:

```text
Python 3.10.20
BMTK 1.2.0
NEURON 8.2.7
```

The structural test run used `pytest 9.1.1`.

Record exact package versions in a public environment file before release.

## Allen reference model

Reference source:

```text
neuronal model ID: 487245719
specimen ID:       486893033
reconstruction ID: 512328466
area/layer:        VISp, Layer 4
model class:       biophysical perisomatic
Cre line:          Scnn1a-Tg3-Cre;Ai14
```

The downloaded model contains:

- `fit_parameters.json`;
- `model_metadata.json`;
- `reconstruction.swc`;
- 16 NMODL mechanism files.

The first test runs this model with its own reconstruction and parameters. At a 0.20 nA somatic current step it produces 16 spikes in the current development environment.

This reference run establishes that the Allen model, BMTK, NEURON and compiled mechanisms work together before MICrONS morphology is introduced.

## Transfer to a MICrONS morphology

The next experiment replaces only the morphology.

The transferred cell uses:

```text
MICrONS processed morphology
+
Allen-derived perisomatic membrane parameters
+
Allen NMODL mechanisms
```

A custom cell processor preserves the complete MICrONS morphology instead of applying the Allen axon replacement.

Model 0:

```text
0.20 nA → no spikes
0.60 nA → 5 spikes
```

The difference from the Allen reference is expected because morphology changes membrane area, passive electrical load and electrotonic structure.

The transfer should not be interpreted as an e-model fit.

## All 20 cells independently

All 20 selected morphologies are instantiated in one BioNet network with no edges between them.

Each receives the same 0.60 nA current step.

Current result:

```text
20/20 finite simulations
20/20 cells spike
122 total spikes during the stimulus window
```

The response range is relatively compact, but this does not imply that the cells truly share the same physiology. It only shows that the common reference parameter set can be simulated on all 20 morphologies.

## Recurrent circuit

The recurrent experiment uses the structural selected-node and recurrent-edge SONATA files.

The edge HDF5 is copied into the run directory so simulation-specific physiology can be added without modifying the structural package.

### Preserved structural information

All 54 contacts preserve:

```text
source node
target node
CAVE synapse ID
afferent section ID
afferent section position
```

### Source stimulation

The stimulated source is selected from the recurrent graph using outgoing connectivity.

Current source:

```text
model 13
```

Current direct targets:

| Target | Contacts from model 13 |
|---|---:|
| 0 | 2 |
| 5 | 1 |
| 6 | 4 |
| 12 | 1 |
| 18 | 1 |

Only model 13 receives direct current.

### Current reference synapse

```text
model:  Exp2Syn
tau1:   0.2 ms
tau2:   1.8 ms
Erev:   0 mV
weight: 0.001 µS
delay:  2.0 ms
```

### Current outcome

The source fires during stimulation.

The direct targets show time-locked depolarisations consistent with the recurrent synapses being active.

No unstimulated cell spikes under the current reference synaptic parameters.

## Next simulation analysis

Before changing synaptic conductance, quantify direct-target responses:

- EPSP amplitude after each source spike;
- latency relative to source spike and configured delay;
- mean/max EPSP by target;
- relationship to number and placement of contacts.

Then perform a predefined sensitivity analysis over a small range of synaptic conductances.

The goal is to show how network recruitment depends on an uncertain modelling parameter, not to increase conductance until a desired propagation pattern appears.

## Generated runs

Simulation directories under `runs/` are generated outputs.
