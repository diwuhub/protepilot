"""
Example: Run the Virtual Analytical QC panel on a sequence.

Usage:
    PYTHONPATH=/path/to/ProtePilot python examples/run_analytical_qc.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytical_qc_twin import run_analytical_qc

# Adalimumab VH+VL
seq = (
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS"
    "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK"
)

result = run_analytical_qc(sequence=seq, pI=8.34, molecule_class="canonical_mab")

print("=== cIEF Charge Variant Profile ===")
print(f"  Acidic: {result.cief.acidic_pct:.1f}%  Main: {result.cief.main_pct:.1f}%  Basic: {result.cief.basic_pct:.1f}%")
print(f"  Sum: {result.cief.acidic_pct + result.cief.main_pct + result.cief.basic_pct:.1f}%")
print(f"  Spec pass: {result.cief.spec_pass}")

print("\n=== CE-SDS Purity ===")
print(f"  Intact: {result.ce_sds.intact_pct:.1f}%  Fragment: {result.ce_sds.fragment_pct:.1f}%  HMW: {result.ce_sds.hmw_pct:.1f}%")

print("\n=== Glycan Profile ===")
print(f"  G0F: {result.glycan.g0f_pct:.1f}%  G1F: {result.glycan.g1f_pct:.1f}%  G2F: {result.glycan.g2f_pct:.1f}%")
print(f"  Man5: {result.glycan.high_mannose_pct:.1f}%  Afuc: {result.glycan.afucosylated_pct:.1f}%")

print(f"\nOverall QC pass: {result.overall_qc_pass}")
