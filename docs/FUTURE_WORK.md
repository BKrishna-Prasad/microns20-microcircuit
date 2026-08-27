# Future work

The current repository reaches a working 20-cell recurrent simulation, but the main scientific objective is still ahead: replacing provisional physiology and missing-input assumptions with models that can be constrained by MICrONS structure and tested against co-registered in-vivo activity.

## 1. Retrieve the matched functional activity

The selected population already preserves the MICrONS functional identifiers needed to recover the corresponding recordings:

```text
(session, scan_idx, unit_id)
```

The next step is to retrieve the matched fluorescence and/or deconvolved activity, together with the timing information required to align it to frames and stimuli.

Two selected neurons currently have multiple functional mappings. These mappings should remain separate until there is a justified rule for combining or selecting them.

## 2. Establish a comparable activity representation

MICrONS measures calcium activity, whereas NEURON produces membrane voltage and spikes. These signals should not be compared directly.

The first practical comparison should use simulated spike activity and MICrONS deconvolved activity after matching the imaging time resolution. If needed, simulated spikes can later be passed through a calcium/indicator observation model and compared in fluorescence space.

## 3. Improve cellular and synaptic physiology

The current 20 cells share one Allen Layer-4 perisomatic parameter set. This is useful for testing the simulation workflow, but it is not a fitted model of the selected MICrONS neurons.

Future cell models should use stronger electrophysiological constraints, for example through Allen/AIBS model families and morphology-aware optimisation with BluePyOpt/BluePyEModel or compatible OBI workflows.

The current recurrent `Exp2Syn` model is also deliberately simple. Later simulations should consider biologically supported synaptic conductance distributions, kinetics, delays, short-term dynamics and uncertainty.

These synaptic parameters describe how a presynaptic event is converted into postsynaptic conductance. They are distinct from the external-input model, which determines the activity arriving from neurons outside the explicitly simulated circuit.

## 4. Characterise the missing input

The selected 20-cell circuit receives:

```text
44,282 incoming contacts
29,991 presynaptic roots outside the selected 20
```

These sources are structurally observed but are not active in the present recurrent simulation.

"Outside the selected 20" is a model-boundary definition, not a biological definition of extrinsic input. The source population may include local V1 neurons, neurons from other cortical areas, inhibitory and excitatory populations, and incompletely reconstructed cells.

The first analysis should therefore classify these roots as far as the available MICrONS annotations allow.

## 5. Build a baseline external-input model

The initial missing-input model should be deliberately low-dimensional and interpretable.

A practical baseline is to provide each selected neuron with fluctuating excitatory and/or inhibitory conductance representing the combined effect of many omitted synapses. The input parameters can be constrained by both:

- structural information, such as the number and class of incoming contacts;
- functional information, such as activity level, temporal variability, autocorrelation and shared population fluctuations.

This baseline should establish whether a simple stochastic drive can place the circuit in an in-vivo-like operating regime before adding thousands of explicit input sources.

The aim is not only to match mean activity. Different input processes can produce similar mean rates, so the temporal and population structure of the recorded activity should also contribute to calibration.

## 6. Refine the input using observed synaptic locations

If the baseline model is stable and the data justify additional complexity, the observed MICrONS input connectivity can be used more directly.

Presynaptic roots outside the explicit 20-cell circuit can be represented as virtual SONATA sources that generate modelled spike trains. Their spikes can activate synaptic mechanisms at the postsynaptic section IDs and positions already mapped from MICrONS.

```text
modelled outside-source activity
        ↓
virtual presynaptic source
        ↓
observed MICrONS contact(s)
        ↓
mapped postsynaptic location
        ↓
20-cell biophysical circuit
```

Multiple contacts belonging to the same presynaptic root should share the same presynaptic activity rather than being treated as independent current injections.

This refinement would allow us to test whether the amount, spatial distribution and temporal statistics of missing input each contribute to the observed network state.

## 7. Calibrate and validate against MICrONS activity

External-input parameters should be calibrated against the co-registered functional activity of the selected neurons.

Candidate models can be compared using quantities such as:

- per-cell activity or event rate;
- temporal variance and autocorrelation;
- pairwise activity relationships;
- population synchrony or shared fluctuations;
- stimulus-dependent activity where appropriate.

Where the recording structure permits it, parameter fitting and validation should be separated. Input parameters can be estimated using one subset of time periods, trials or stimulus epochs and evaluated on held-out activity.

A successful model should therefore reproduce more than a population-average firing rate. It should provide a plausible operating state while remaining consistent with the observed structural amount and location of missing input.

## 8. Generalise the workflow

Once the 20-cell model and validation strategy are stable, the same workflow can be tested on another simultaneously recorded population or a moderately larger circuit.

The intended reusable outcome is the calibration procedure rather than one fixed parameter set:

```text
MICrONS structure
        +
co-registered activity
        ↓
external-input calibration
        ↓
OBI-compatible simulation
        ↓
held-out functional validation
```

This would make it possible to test whether inferred input statistics generalise across neurons, recordings and larger MICrONS-derived circuits.

## Software development

Before a stable release, the simulation code should also be simplified and tested further, including dedicated tests for BMTK compatibility, synapse preservation, input generation and simulation outputs.
