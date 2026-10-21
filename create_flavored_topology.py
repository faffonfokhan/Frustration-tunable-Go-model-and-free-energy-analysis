#!/usr/bin/env python3
"""
COMPLETE FOOLPROOF script to add MJ non-native interactions to Go model.
This converts single CA type to flavored types and adds all MJ interactions.

Parameters: f = 0.02, sigma = 0.4 nm, Temperature = 115 K
Author: For CI2 protein folding simulation
"""

import pandas as pd
import numpy as np
import sys
from collections import OrderedDict

# ============================================================================
# PARAMETERS - EXACTLY AS SPECIFIED
# ============================================================================
SCALING_FACTOR = 0.60# f = 0.05
SIGMA = 0.4  # nm - EXACTLY 0.4
SIGMA_6 = SIGMA**6  # 0.004096
SIGMA_12 = SIGMA**12  # 1.67772e-05

# Amino acid code mapping
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def read_mj_matrix(csv_file):
    """Read normalized MJ matrix from CSV."""
    df = pd.read_csv(csv_file, index_col=0)
    print(f"✓ MJ matrix loaded: {df.shape[0]} × {df.shape[1]}")
    return df

def calculate_c6_c12(epsilon_nn):
    """
    Calculate C6 and C12 for 6-12 Lennard-Jones potential.
    V(r) = C12/r^12 - C6/r^6
    
    Standard LJ form: V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
    Therefore:
        C12 = 4 * epsilon * sigma^12
        C6  = 4 * epsilon * sigma^6
    """
    c6 = 4.0 * epsilon_nn * SIGMA_6
    c12 = 4.0 * epsilon_nn * SIGMA_12
    return c6, c12

def parse_topology(top_file):
    """Parse the entire topology file into sections."""
    sections = OrderedDict()
    current_section = None
    section_lines = []
    
    with open(top_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            
            # Detect section headers
            if stripped.startswith('[') and stripped.endswith(']'):
                # Save previous section
                if current_section is not None:
                    sections[current_section] = section_lines
                
                # Start new section
                current_section = stripped[1:-1].strip()
                section_lines = []
            else:
                section_lines.append(line)
    
    # Save last section
    if current_section is not None:
        sections[current_section] = section_lines
    
    return sections

def extract_residue_sequence(atoms_lines):
    """Extract residue sequence from [ atoms ] section."""
    residues = []
    
    for line in atoms_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        
        parts = stripped.split()
        if len(parts) >= 7:
            atom_nr = parts[0]
            atom_type = parts[1]
            res_nr = parts[2]
            res_name = parts[3]
            atom_name = parts[4]
            
            if res_name in AA_3_TO_1:
                residues.append({
                    'nr': atom_nr,
                    'old_type': atom_type,
                    'res_nr': res_nr,
                    'res_name': res_name,
                    'atom_name': atom_name,
                    'aa_1letter': AA_3_TO_1[res_name],
                    'cgnr': parts[5],
                    'charge': parts[6],
                    'mass': parts[7] if len(parts) > 7 else '1.000'
                })
    
    print(f"✓ Found {len(residues)} CA atoms in sequence")
    return residues

def get_unique_residue_types(residues):
    """Get unique residue types present in the protein."""
    unique = OrderedDict()
    for res in residues:
        res_name = res['res_name']
        if res_name not in unique:
            unique[res_name] = res['aa_1letter']
    
    print(f"✓ Unique residue types: {len(unique)}")
    for res_name, aa_code in unique.items():
        print(f"    {res_name} = {aa_code}")
    
    return unique

def create_atomtypes_section(unique_residues):
    """Create [ atomtypes ] section with flavored types."""
    lines = []
    lines.append(" ;name  mass     charge   ptype c10       c12\n")
    
    for res_name in unique_residues.keys():
        flavored_type = f"CA_{res_name}"
        lines.append(f" {flavored_type:<6s} 1.000    0.000 A    0.000   0.167772160E-04\n")
    
    return lines

def create_atoms_section(residues):
    """Create [ atoms ] section with flavored atom types."""
    lines = []
    lines.append(" ;nr  type  resnr residue atom  cgnr charge  mass\n")
    
    for res in residues:
        flavored_type = f"CA_{res['res_name']}"
        line = f"{res['nr']:>6s}  {flavored_type:<8s} {res['res_nr']:>4s}  {res['res_name']:>3s}  "
        line += f"{res['atom_name']:<4s} {res['cgnr']:>4s}   {res['charge']:>6s}   {res['mass']:>6s}\n"
        lines.append(line)
    
    return lines

def create_nonbond_params_section(unique_residues, mj_matrix):
    """
    Create [ nonbond_params ] section with ALL MJ-based interactions.
    This creates ALL pairwise interactions between residue types.
    """
    lines = []
    lines.append("; Non-native interactions based on Miyazawa-Jernigan matrix\n")
    lines.append(f"; Scaling factor: f = {SCALING_FACTOR}\n")
    lines.append(f"; Sigma: σ = {SIGMA} nm\n")
    lines.append("; Potential: V(r) = C12/r^12 - C6/r^6 (6-12 Lennard-Jones)\n")
    lines.append("; C6 = 4 × ε_nn × σ^6\n")
    lines.append("; C12 = 4 × ε_nn × σ^12\n")
    lines.append("; Format: type_i  type_j  func  C6  C12\n")
    lines.append(";\n")
    
    residue_types = list(unique_residues.keys())
    pair_count = 0
    
    # Generate ALL pairs (i <= j to avoid duplicates)
    for i, res_i in enumerate(residue_types):
        for res_j in residue_types[i:]:
            aa_i = unique_residues[res_i]
            aa_j = unique_residues[res_j]
            
            # Get MJ energy from matrix
            if aa_i in mj_matrix.columns and aa_j in mj_matrix.index:
                mj_value = float(mj_matrix.loc[aa_j, aa_i])
                
                # Calculate epsilon_nn
                epsilon_nn = SCALING_FACTOR * abs(mj_value)
                
                # Calculate C6 and C12
                c6, c12 = calculate_c6_c12(epsilon_nn)
                
                # Create atom type names
                type_i = f"CA_{res_i}"
                type_j = f"CA_{res_j}"
                
                # Write line
                line = f"{type_i:<10s} {type_j:<10s}  1  "
                line += f"{c6:.6e}  {c12:.6e}  "
                line += f"; {aa_i}-{aa_j} (MJ={mj_value:.3f}, ε={epsilon_nn:.4f} kJ/mol)\n"
                lines.append(line)
                
                pair_count += 1
    
    lines.append(f";\n; Total pairs: {pair_count}\n")
    print(f"✓ Generated {pair_count} non-native interaction pairs")
    
    return lines

def write_modified_topology(sections, output_file, new_atomtypes, new_atoms, nonbond_params):
    """
    Write the complete modified topology file with sections in the correct order.
    """
    
    # Define the GROMACS-mandated order of sections
    # We will insert [ nonbond_params ] after [ atomtypes ]
    section_order = [
        'defaults',
        'atomtypes',
        # --- Our insertion point ---
        'moleculetype',
        'atoms',
        'bonds',
        'pairs',
        'exclusions',
        'angles',
        'dihedrals',
        'system',
        'molecules'
    ]

    with open(output_file, 'w') as f:
        for section_name in section_order:
            # Skip sections that don't exist in the input file
            if section_name not in sections:
                continue

            # Write the section header
            f.write(f" [ {section_name} ]\n")
            
            # --- WRITE THE CONTENT FOR THIS SECTION ---
            
            if section_name == 'atomtypes':
                f.writelines(new_atomtypes)
                # --- THIS IS THE FIX ---
                # After writing the new atomtypes, immediately write the
                # nonbond_params section. This places it in the correct location.
                f.write("\n\n")  # Add some space for readability
                f.write(" [ nonbond_params ]\n")
                f.writelines(nonbond_params)
                # ----------------------

            elif section_name == 'atoms':
                f.writelines(new_atoms)
            else:
                # For all other sections, write the original content
                f.writelines(sections[section_name])

            f.write("\n")
    
    print(f"✓ Modified topology written correctly to: {output_file}")

def generate_table_xvg(filename='table.xvg', r_max=2.0, n_points=2001):
    """
    Generate 6-12 Lennard-Jones table for GROMACS.
    Used with: gmx mdrun -table table.xvg
    """
    r = np.linspace(0, r_max, n_points)
    r[0] = 1e-10  # Avoid division by zero
    
    # Calculate LJ terms
    r6 = r**6
    r7 = r**7
    r12 = r**12
    r13 = r**13
    
    # V(r) = 1/r^12 - 1/r^6 (normalized form)
    # -dV/dr = 12/r^13 - 6/r^7
    
    f = 1.0/r12 - 1.0/r6
    minus_fp = 12.0/r13 - 6.0/r7
    
    with open(filename, 'w') as out:
        out.write("# GROMACS table for 6-12 Lennard-Jones potential\n")
        out.write("# For non-native MJ interactions in flavored Go model\n")
        out.write("# V(r) = C12/r^12 - C6/r^6\n")
        out.write("# Generated for f=0.05, sigma=0.4 nm\n")
        out.write("#\n")
        
        for i in range(n_points):
            # Format: r  f(r)  -f'(r)  g(r)  -g'(r)  h(r)  -h'(r)
            # For pure LJ, Coulomb (f) = 0, use combined LJ
            out.write(f"{r[i]:15.10e} {f[i]:15.10e} {minus_fp[i]:15.10e} ")
            out.write(f"{0.0:15.10e} {0.0:15.10e} {0.0:15.10e} {0.0:15.10e}\n")
    
    print(f"✓ Table file created: {filename}")

def generate_tablep_xvg(filename='tablep.xvg', r_max=2.0, n_points=2001):
    """
    Generate pair interaction table for GROMACS.
    Used with: gmx mdrun -tablep tablep.xvg
    Specifically for [ nonbond_params ] interactions.
    """
    r = np.linspace(0, r_max, n_points)
    r[0] = 1e-10
    
    r6 = r**6
    r7 = r**7
    r12 = r**12
    r13 = r**13
    
    f = 1.0/r12 - 1.0/r6
    minus_fp = 12.0/r13 - 6.0/r7
    
    with open(filename, 'w') as out:
        out.write("# GROMACS pair interaction table\n")
        out.write("# For [ nonbond_params ] with func=1\n")
        out.write("# 6-12 Lennard-Jones: V(r) = C12/r^12 - C6/r^6\n")
        out.write("#\n")
        
        for i in range(n_points):
            # Format: r  f(r)  -f'(r)
            out.write(f"{r[i]:15.10e} {f[i]:15.10e} {minus_fp[i]:15.10e}\n")
    
    print(f"✓ Pair table created: {filename}")

def print_example_calculations(mj_matrix):
    """Print example calculations for verification."""
    
    print("\n" + "="*80)
    print("EXAMPLE CALCULATIONS")
    print("="*80)
    print(f"Scaling factor f = {SCALING_FACTOR}")
    print(f"Sigma σ = {SIGMA} nm")
    print(f"σ^6  = {SIGMA_6:.6e}")
    print(f"σ^12 = {SIGMA_12:.6e}")
    print("="*80)
    
    examples = [
        ('G', 'G', 'Gly-Gly'),
        ('W', 'W', 'Trp-Trp'),
        ('F', 'F', 'Phe-Phe'),
        ('L', 'L', 'Leu-Leu'),
        ('K', 'K', 'Lys-Lys'),
        ('N', 'N', 'Asn-Asn'),
    ]
    
    for aa1, aa2, name in examples:
        if aa1 in mj_matrix.columns and aa2 in mj_matrix.index:
            mj_val = float(mj_matrix.loc[aa2, aa1])
            epsilon_nn = SCALING_FACTOR * abs(mj_val)
            c6, c12 = calculate_c6_c12(epsilon_nn)
            
            print(f"\n{name}:")
            print(f"  MJ matrix value = {mj_val:.4f}")
            print(f"  ε_nn = {SCALING_FACTOR} × {abs(mj_val):.4f} = {epsilon_nn:.6f} kJ/mol")
            print(f"  C6  = 4 × {epsilon_nn:.6f} × {SIGMA_6:.6e} = {c6:.6e} kJ·nm^6/mol")
            print(f"  C12 = 4 × {epsilon_nn:.6f} × {SIGMA_12:.6e} = {c12:.6e} kJ·nm^12/mol")
    
    print("\n" + "="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("FLAVORED GO MODEL WITH MJ NON-NATIVE INTERACTIONS")
    print("="*80)
    print("This script will:")
    print("  1. Convert your topology from single CA type to flavored types")
    print("  2. Add ALL pairwise MJ non-native interactions")
    print("  3. Generate GROMACS table files for 6-12 LJ potential")
    print("="*80)
    
    if len(sys.argv) < 3:
        print("\nUSAGE:")
        print(f"  python {sys.argv[0]} input.top mj_matrix.csv [output.top]")
        print("\nEXAMPLE:")
        print(f"  python {sys.argv[0]} gromacs_topology_file.top mj_matrix_normalised.csv flavored_MJ.top")
        print("="*80 + "\n")
        sys.exit(1)
    
    input_top = sys.argv[1]
    mj_csv = sys.argv[2]
    output_top = sys.argv[3] if len(sys.argv) > 3 else 'flavored_topology_MJ.top'
    
    print(f"\nInput topology:  {input_top}")
    print(f"MJ matrix:       {mj_csv}")
    print(f"Output topology: {output_top}")
    print(f"Parameters:      f = {SCALING_FACTOR}, σ = {SIGMA} nm")
    print("="*80 + "\n")
    
    # Step 1: Read MJ matrix
    print("Step 1: Reading MJ matrix...")
    mj_matrix = read_mj_matrix(mj_csv)
    
    # Step 2: Parse topology
    print("\nStep 2: Parsing topology file...")
    sections = parse_topology(input_top)
    
    # Step 3: Extract residue sequence
    print("\nStep 3: Extracting residue sequence...")
    residues = extract_residue_sequence(sections['atoms'])
    
    # Step 4: Get unique residue types
    print("\nStep 4: Identifying unique residue types...")
    unique_residues = get_unique_residue_types(residues)
    
    # Step 5: Create new sections
    print("\nStep 5: Creating flavored [ atomtypes ] section...")
    new_atomtypes = create_atomtypes_section(unique_residues)
    
    print("Step 6: Creating flavored [ atoms ] section...")
    new_atoms = create_atoms_section(residues)
    
    print("Step 7: Creating [ nonbond_params ] section with MJ interactions...")
    nonbond_params = create_nonbond_params_section(unique_residues, mj_matrix)
    
    # Step 6: Write modified topology
    print("\nStep 8: Writing modified topology file...")
    write_modified_topology(sections, output_top, new_atomtypes, new_atoms, nonbond_params)
    
    # Step 7: Generate table files
    print("\nStep 9: Generating GROMACS table files...")
    generate_table_xvg('table.xvg')
    generate_tablep_xvg('tablep.xvg')
    
    # Step 8: Print examples
    print_example_calculations(mj_matrix)
    
    print("="*80)
    print("SUCCESS! All files generated:")
    print(f"  1. {output_top} - Flavored topology with MJ interactions")
    print(f"  2. table.xvg - 6-12 LJ table")
    print(f"  3. tablep.xvg - Pair interaction table")
    print("\nTo run GROMACS simulation:")
    print(f"  gmx grompp -f md.mdp -c structure.gro -p {output_top} -o md.tpr")
    print(f"  gmx mdrun -deffnm md -table table.xvg -tablep tablep.xvg")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
