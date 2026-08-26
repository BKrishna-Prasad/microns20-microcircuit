# Known limitations and planned cleanup

This file tracks limitations of the current repository without mixing them with older development history.

## Scientific limitations

- The selected cells do not yet have cell-specific fitted e-models.
- One Allen Layer-4 perisomatic parameter set is currently reused across all 20 cells.
- The transferred parameters were fitted on a different morphology.
- The current recurrent synaptic model is a simple reference `Exp2Syn`.
- Synaptic conductance, delay and kinetics are not measured for the 54 contacts.
- The current recurrent simulation does not include the 44,282 outside-selected20 incoming contacts.
- Functional calcium/activity traces have not yet been retrieved into the processed dataset.
- The current network activity has therefore not been validated against MICrONS functional recordings.
- Two recurrent contacts map to axonal target sections and need an additional source-annotation check.
- The 20 current skeletons contain no separate type-4 apical labels.

## Software cleanup

### Simulation modules

Current simulation code was developed incrementally:

```text
simulation.py
cell_processors.py
population_simulation.py
recurrent_simulation.py
```

Before a stable release, consolidate common functionality and isolate simulator-specific code.

### BMTK 1.2 compatibility

Move the two current run-local adapters into a dedicated module and add automated tests.

### Duplicate notebook figures

Some notebook plotting calls render figures twice because a figure object is both displayed by the plotting function/library and returned as the final cell expression.

Standardise plotting functions so each figure is displayed once.

### Configuration coverage

Review every configuration field and confirm that changing it changes the corresponding implementation.

### Simulation tests

The structural tests pass, but simulation utilities need their own tests.

