"""Public model API for the proposed Risk-Aware Sparse Volumetric Transformer.

    model = RSVT(dimension=96, heads=4, layers=2, sequence_cap=128)
    model.train()
    output = model(image_B1ZYX, valid_B1ZYX, voxel_xyz_to_LPS_B44)

Training uses dense soft gates; eval() uses genuinely packed hard selection.
Mandatory coarse tokens always remain. Classification is TB/TM/FB/FM.
The spatial head defaults off because existing point annotations are not masks.
Theoretical rationale and implementation mapping: docs/rsvt_theory.md.
"""

from rsvt.models.risk_rsvt import RSVTRiskFoundation


class RSVT(RSVTRiskFoundation):
    def __init__(self, *, localization_head=False, **configuration):
        super().__init__(localization_head=localization_head, **configuration)
