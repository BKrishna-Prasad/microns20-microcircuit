# Project overview

## Why this repository exists

The target research problem is to build a small, openly reproducible MICrONS microcircuit that can be simulated and compared with the functional measurements associated with the same neurons. A central unresolved part of that problem is the input arriving from neurons outside the explicitly modelled circuit.

The repository therefore has two connected goals:

1. construct a traceable structural and simulation-ready 20-neuron MICrONS circuit; and
2. provide the software base needed to study how external drive must be modelled to reproduce the observed activity.

The current repository has reached the first goal at a preliminary simulation level. The second goal remains future work.

## Current scientific question

For the present implementation we asked:

> Can we start from MICrONS public data, select a simultaneously recorded and structurally connected set of neurons, preserve their CAVE-derived morphology and synaptic locations, represent the circuit in SONATA, and run it in BioNet/NEURON without losing the structural information?

The answer is yes for the current 20-cell circuit.

## Selected circuit

The current selection contains 20 excitatory neurons from MICrONS `session 9 / scan 3`.

The recording was not entered manually. Feasible recordings were compared using the number of eligible cells and structural-connectivity statistics. The selected recording contained 29 morphology-eligible candidates. The final 20-cell subset was then selected by connectivity optimisation.

The resulting subnetwork contains:

- 20 neurons;
- 9 L4a and 11 L4b cells;
- 38 directed recurrent pairs;
- 54 individual recurrent synaptic contacts;
- 22 functional mappings, because two selected neurons have more than one preserved functional unit mapping.

## Structural representation

Each selected neuron is represented by a processed SWC derived from the corresponding MICrONS CAVE skeleton.

The processed morphologies preserve:

- CAVE geometry;
- soma, axon and dendrite labels;
- available apical labels;
- topology;
- the global MICrONS coordinate frame.

The current 20 cells contain no type-4 apical labels in their source skeletons. This should not be interpreted as proof that the biological neurons have no apical dendritic structure; it means that a distinct type-4 compartment was not present in these skeleton annotations.

The circuit also retains every observed selected-to-selected synapse as a separate contact. Each contact is mapped to a postsynaptic NEURON/BMTK section and position.

## Preliminary electrical model

MICrONS functional recordings are calcium imaging, not patch-clamp recordings of these exact neurons. We therefore do not yet have cell-specific electrophysiological targets for fitting ion-channel conductances.

To establish that the circuit can run, we used Allen Cell Types neuronal model `487245719` (specimen `486893033`), a VISp Layer-4 `Scnn1a-Tg3-Cre` biophysical perisomatic model, as a reference parameter source.

We first ran the Allen model with its own morphology and mechanisms. We then transferred the same parameter set onto one MICrONS morphology while preserving the MICrONS axon, followed by all 20 morphologies.

This transfer is intentionally described as **Allen-derived reference physiology**, not as a fitted MICrONS e-model.

## Preliminary recurrent simulation

The current connected experiment uses:

- all 20 processed MICrONS morphologies;
- all 54 observed recurrent contacts;
- the saved postsynaptic section IDs and section positions;
- one shared Allen-derived perisomatic parameter set;
- a simple excitatory NEURON `Exp2Syn` model;
- one stimulated source neuron chosen from the recurrent graph.

The first reference synapse parameters are:

```text
tau1 = 0.2 ms
tau2 = 1.8 ms
Erev = 0 mV
weight = 0.001 µS per anatomical contact
delay = 2.0 ms
```

These are simulation reference parameters, not measurements of the 54 MICrONS synapses.

In the present recurrent run:

- all 20 voltage traces are finite;
- model 13 is selected as the stimulation source by outgoing connectivity;
- model 13 has five direct targets: 0, 5, 6, 12 and 18;
- the source fires during the current step;
- direct targets show postsynaptic depolarisations;
- none of the unstimulated cells spikes with the current reference synaptic parameters.

## What this result does and does not show

The current work demonstrates that the structural MICrONS circuit can be carried through to a working multicompartment recurrent simulation.

It does not yet show that the simulated firing rates, spike shapes, synaptic responses or network state match the recorded MICrONS biology.

That validation requires the later stages described in `FUTURE_WORK.md`.
