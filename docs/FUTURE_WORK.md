# Future work

The current repository reaches a working recurrent simulation, but the main scientific problem is still ahead: replacing provisional physiology and input assumptions with models that can be constrained and tested against the MICrONS functional data.

## 1. Retrieve the matched functional activity

The selected population already contains the functional identifiers required to locate the corresponding recordings.

The current plan is to use MICrONS NDA v8 and preserve the exact functional key for every mapping:

```text
(session, scan_idx, unit_id)
```

Relevant NDA tables include:

- `ScanUnit` for unit identity within a scan;
- `Fluorescence` for raw fluorescence;
- `Activity` for deconvolved activity;
- timing/scan information needed to align activity to frames and stimuli.

Two selected neurons have multiple mappings. Those mappings should remain separate until there is a justified rule for combining or selecting them.

The first functional-data deliverable should be a table that proves that every retained mapping can be retrieved and aligned to a time base.

Suggested location:

```text
data/processed/functional/
```

This functional stream should not change the structural selection.

## 2. Establish comparable activity representations

The observed data are calcium imaging, whereas NEURON produces membrane voltage and spikes.

Directly fitting ion-channel conductances to calcium traces alone is underdetermined.

A sensible comparison requires an explicit observation model. Options include:

- compare deconvolved MICrONS activity with simulated spike trains after matching temporal resolution;
- convert simulated spikes through a calcium/indicator response model and compare in fluorescence space;
- use both representations as complementary checks.

This layer should be explicit so a mismatch is not hidden by preprocessing.

## 3. Replace the shared Allen reference parameters

The current 20 cells all use one Allen Layer-4 perisomatic parameter set.

Future cell models should use stronger biological constraints, for example:

- Allen/AIBS electrophysiology-based model families matched by layer and cell class;
- BluePyOpt/BluePyEModel optimisation workflows;
- available OBI/Blue Brain model recipes where compatible;
- morphology-aware re-optimisation rather than direct transfer of conductances fitted on another morphology.

A key issue is that the MICrONS cells themselves do not provide patch-clamp targets. Therefore the optimisation problem should separate:

```text
intrinsic electrophysiological priors
from
network/activity calibration against calcium imaging
```

rather than asking calcium data to identify every ion-channel parameter.

## 4. Improve synaptic physiology

The current recurrent `Exp2Syn` model is deliberately simple.

Future work should consider:

- excitatory postsynaptic conductance distributions;
- connection-specific delays;
- synapse-size information;
- short-term plasticity if justified;
- multiple contacts between the same pre/post pair;
- dendritic location effects;
- uncertainty rather than one deterministic conductance value.

The two current axon-target recurrent contacts should also be cross-checked against source target-structure annotations.

## 5. Classify outside-selected20 input

The current circuit observes:

```text
44,282 incoming contacts
29,991 positive presynaptic roots
```

from outside the selected 20.

These inputs should be separated into biologically meaningful classes where possible, for example:

- local V1 neurons outside the explicit 20-cell model;
- other cortical areas;
- inhibitory versus excitatory sources where annotations support this;
- sources with/without soma or cell-type information;
- longer-range or otherwise incomplete presynaptic reconstructions.

This classification is essential because "outside selected 20" is a model-boundary definition, not a biological definition of extrinsic input.

## 6. Model the external drive

Once input classes are understood, the main modelling task is to define activity for the sources that are not explicitly simulated as detailed neurons.

Possible approaches include:

- spike trains derived from measured functional activity when available;
- population-statistical spike generators;
- inhomogeneous Poisson or renewal processes as baselines;
- low-dimensional latent drive shared across source classes;
- stimulus-conditioned input models;
- hierarchical models in which external-drive parameters are inferred jointly with network-state parameters.

The 44,282 observed incoming synapse locations provide a valuable structural constraint even when the presynaptic dynamics are not directly observed.

## 7. Calibrate the network state

The present recurrent network has no background drive from the outside-selected20 population.

That means the unstimulated cells begin near their passive/resting state, which is not intended to represent the in-vivo operating point.

Future calibration should consider:

- baseline firing/activity level;
- trial-to-trial variability;
- response reliability;
- pairwise or population activity relationships;
- stimulus dependence;
- calcium-imaging timescale.

The external-input model is likely to be central to reproducing the in-vivo network state.

## 8. Validate against MICrONS recordings

Validation should be held out from parameter tuning where possible.

Useful targets may include:

- per-cell activity rate;
- event probability;
- pairwise functional relationships;
- population synchrony;
- response to repeated stimuli;
- temporal autocorrelation;
- network-level activity distributions.

The exact validation targets should be chosen after the NDA activity is inspected.

## 9. Expand beyond 20 neurons

The current 20-cell model is deliberately small enough to inspect and simulate easily.

The configuration should make it possible to rerun with:

- another recording;
- a different population size;
- broader proofreading criteria;
- different cell-class restrictions.

Scaling should happen after the model and validation strategy is stable, not simply because more neurons are available.

## 10. Software work

Before a stable release:

- consolidate duplicated simulation utilities;
- isolate BMTK 1.2 compatibility code;
- add automated simulation-format tests;
- stop duplicate figure rendering in notebooks;
- make every documented configuration switch effective in the implementation;
- remove stale helpers from earlier development;
- add a reproducible environment specification and CI;
- keep public outputs small and regenerate large run directories.
