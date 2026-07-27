import numpy as np

def validate_telemetry_gate(soft_flux, hard_flux, threshold_drop=1e-9):
    """
    Programmatic context-aware gate. Filters out telemetry drops,
    missing values, and singular instrument particle spikes.
    """
    # Rule 1: Handle physical telemetry dropouts/gaps
    if soft_flux <= threshold_drop or hard_flux <= threshold_drop:
        return "DROP", False
        
    # Rule 2: Handle data variance anomalies (NaN records)
    if np.isnan(soft_flux) or np.isnan(hard_flux):
        return "NAN_ERROR", False
        
    # Rule 3: Context-Aware Validation (Isolate cosmic ray particle hits)
    # True flares force a correlated rise across BOTH energy channels.
    # A massive spike in Hard X-rays WITHOUT a matching Soft X-ray shift = Noise.
    if hard_flux > 1e-5 and soft_flux < 1e-6:
        return "PARTICLE_SPIKE_NOISE", False
        
    return "VALID_SIGNAL", True
