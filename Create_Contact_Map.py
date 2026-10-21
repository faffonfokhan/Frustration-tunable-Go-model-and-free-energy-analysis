import MDAnalysis as mda
import numpy as np
import math
import csv
import argparse
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate and visualize contact matrix from trajectory')
    parser.add_argument('-top', required=True, help='Topology file (.top)')
    parser.add_argument('-xtc', required=True, help='Trajectory file (.xtc)')
    parser.add_argument('-pdb', required=True, help='PDB structure file for reference (.pdb)')
    parser.add_argument('-o', default='CM.csv', help='Output file name (default: CM.csv)')
    parser.add_argument('-beta', type=float, default=5.0, help='Beta parameter (default: 5.0)')
    parser.add_argument('-viz', action='store_true', help='Generate visualization')
    parser.add_argument('-vmin', type=float, default=0.0, help='Minimum value for heatmap (default: 0.0)')
    parser.add_argument('-vmax', type=float, default=1.0, help='Maximum value for heatmap (default: 1.0)')
    parser.add_argument('-cmap', default='jet', help='Colormap for visualization (default: jet)')
    parser.add_argument('-nocont', action='store_true', help='Do not show native contacts in visualization')
    parser.add_argument('-Q_min', type=float, default=0.0, help='Minimum Q value for frame filtering (default: 0.0)')
    parser.add_argument('-Q_max', type=float, default=1.0, help='Maximum Q value for frame filtering (default: 1.0)')
    return parser.parse_args()


def extract_pairs_from_top(top_file):
    pairs = []
    in_pairs_section = False
    
    with open(top_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '[ pairs ]':
                in_pairs_section = True
                continue
            if in_pairs_section and line.startswith('['):
                break
            if in_pairs_section and line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append(parts)
    
    pairs_filepath = 'pairs.txt'
    with open(pairs_filepath, 'w') as file:
        for pair in pairs:
            file.write(' '.join(pair) + '\n')
    
    return pairs_filepath


def calculate_con_matrix(contact, Fin_con, atom_to_residue, n_residues):
    CON = np.zeros((n_residues + 1, n_residues + 1))
    contact_count = np.zeros((n_residues + 1, n_residues + 1))
    
    for i in range(len(contact[0])):
        atom1_idx = contact[0][i]
        atom2_idx = contact[1][i]

        residue1 = atom_to_residue.get(atom1_idx)
        residue2 = atom_to_residue.get(atom2_idx)
        
        if residue1 is None or residue2 is None:
            print(f"Warning: Could not find residue for atom index {atom1_idx} or {atom2_idx}. Skipping contact.")
            continue

        CON[residue1][residue2] += Fin_con[i]
        contact_count[residue1][residue2] += 1
        
        if residue1 != residue2:
            CON[residue2][residue1] += Fin_con[i]
            contact_count[residue2][residue1] += 1
    
    contact_count[contact_count == 0] = 1
    CON /= contact_count
    
    return CON


def plot_contact_map(con_matrix, output_file, title="Contact Map", vmin=0, vmax=1, cmap='jet', show_native_contacts=False, native_contacts=None):
    plt.figure(dpi=120, figsize=(8, 6))
    sns.heatmap(data=con_matrix, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap)).invert_yaxis()
    plt.title(title)
    
    if show_native_contacts and native_contacts is not None:
        resi1, resi2 = native_contacts
        plt.scatter(resi1, resi2, marker='o', color='none', edgecolors='white', s=40)
   
    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.grid(color='black', linewidth=0.1)
    plt.savefig(output_file)
    plt.close()
    print(f"Contact map visualization saved to {output_file}")


def preprocess_native_contacts(pdb_file):
    n1, n2 = [], []
    resi1, resi2 = [], []
    
    try:
        with open("contact.csv") as con:
            for line in con:
                cols = line.split()
                if len(cols) >= 2:
                    n1.append(float(cols[0]))
                    n2.append(float(cols[1]))
    except FileNotFoundError:
        print("Warning: contact.csv not found. Native contacts will not be plotted.")
        return None
    
    if not n1:
        return None

    for i in range(len(n1)):
        resi1.append(float(n1[i] - 0.5))
        resi2.append(float(n2[i] - 0.5)) 
    return [resi1, resi2]


def main():
    args = parse_args()
    try:
        u = mda.Universe(args.pdb, args.xtc)
    except Exception as e:
        print(f"Error creating MDAnalysis Universe: {e}")
        print("Please ensure your -pdb and -xtc files are correct and compatible.")
        return
    atom_to_residue = {atom.index + 1: atom.resid for atom in u.atoms}
    n_residues = len(u.residues)
    ATOMNUM = len(u.atoms)
    
    print(f"Successfully loaded Universe. Number of atoms: {ATOMNUM}, Number of residues: {n_residues}")

    pairs_file = extract_pairs_from_top(args.top)
    contact = []
    CONNUM = 0
    with open(pairs_file, 'r') as file:
        for line in file:
            if line.startswith(';') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                atom1_idx = int(parts[0])
                atom2_idx = int(parts[1])

                try:
                    atom1 = u.atoms[atom1_idx - 1] # Convert to 0-based index
                    atom2 = u.atoms[atom2_idx - 1] # Convert to 0-based index
                except IndexError:
                    print(f"Fatal Error: Atom index {atom1_idx} or {atom2_idx} from '{args.top}' is out of bounds for the {ATOMNUM} atoms in your structure.")
                    return

                distance = np.linalg.norm(atom1.position - atom2.position)
                contact.append([atom1_idx, atom2_idx, distance])
                CONNUM += 1
    
    if CONNUM == 0:
        print("Fatal Error: No contacts found in the [ pairs ] section of the topology file.")
        return

    contact = list(map(list, zip(*contact)))
    print(f"Number of native contacts parsed: {CONNUM}")
    
    conjs = np.zeros(CONNUM)
    total_frames = 0
    valid_frames = 0
    
    Q_min = args.Q_min
    Q_max = args.Q_max
    print(f"Filtering frames by normalized Q value: {Q_min} <= Q <= {Q_max}")
    
    beta = args.beta
    EXP_THRESHOLD = 700

    for ts in u.trajectory:
        total_frames += 1
        Qi = 0
        temp_conjs = np.zeros(CONNUM)
        
        for i in range(CONNUM):
            a1_idx = contact[0][i] - 1 # 0-indexed for MDAnalysis
            a2_idx = contact[1][i] - 1 # 0-indexed for MDAnalysis
            r0 = contact[2][i]
            
            atom1 = u.atoms[a1_idx]
            atom2 = u.atoms[a2_idx]
            distance = np.linalg.norm(atom1.position - atom2.position)
            
            exp_argument = beta * (distance - 1.2 * r0)
            
            qi = 1.0 / (1.0 + math.exp(exp_argument)) if exp_argument < EXP_THRESHOLD else 0
            
            temp_conjs[i] = qi
            Qi += qi
        
        normalized_Q = Qi / CONNUM
        
        if Q_min <= normalized_Q <= Q_max:
            conjs += temp_conjs
            valid_frames += 1
            
    if valid_frames > 0:
        Fin_con = [val / valid_frames for val in conjs]
        print(f"Used {valid_frames} out of {total_frames} frames.")
    else:
        Fin_con = [0.0] * CONNUM
        print(f"WARNING: No frames found in the specified Q value range.")
    
    CON = calculate_con_matrix(contact, Fin_con, atom_to_residue, n_residues)
    
    output_file = args.o
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(CON)
    print(f"Contact matrix saved to {output_file}")
    
    if args.viz:
        native_contacts = preprocess_native_contacts(args.pdb)
        viz_output = os.path.splitext(output_file)[0] + ".png"
        plot_contact_map(
            CON, 
            viz_output,
            title=f"Contact Map (Q: {Q_min}-{Q_max}, {valid_frames}/{total_frames} frames)",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            show_native_contacts=(not args.nocont and native_contacts is not None),
            native_contacts=native_contacts
        )


if __name__ == "__main__":
    main()
